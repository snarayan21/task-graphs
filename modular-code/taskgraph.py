import numpy as np
from pydrake.solvers.mathematicalprogram import MathematicalProgram
from pydrake.solvers.mathematicalprogram import Solve
import pydrake.math as math
import matplotlib.pyplot as plt
import networkx as nx
from reward_oracle import RewardOracle
from scipy.stats import norm
import os


class TaskGraph:
    # class for task graphs where nodes are tasks and edges are precedence relationships

    def __init__(self, num_tasks, edges, coalition_params, coalition_types, dependency_params, dependency_types, aggs,
                 numrobots):
        self.num_tasks = num_tasks
        self.num_robots = numrobots
        self.edges = edges
        self.task_graph = nx.DiGraph()
        self.task_graph.add_nodes_from(range(num_tasks))
        self.task_graph.add_edges_from(edges)
        self.num_edges = len(edges)  # number of edges

        self.coalition_params = coalition_params
        self.coalition_types = coalition_types
        self.dependency_params = dependency_params
        self.dependency_types = dependency_types
        self.fig = None
        self.influence_agg_func_types = aggs
        self.reward_distributions = [None for _ in range(self.num_tasks)]

        # variables using in the optimization
        self.var_flow = None

        # variables used during run-time
        self.flow = None
        self.reward = np.zeros(self.num_tasks)
        self.reward_mean = np.zeros(self.num_tasks)
        self.reward_variance = np.zeros(self.num_tasks)

    def identity(self, f):
        """
        Identity function (for passing into pydrake)
        :return:
        """
        return f

    def compute_node_coalition(self, node_i, f):
        """
        This function computes the coalition function output for a given node and a given flow
        :param node_i, shape=(1x1), index of node for which coalition function has to be evaluated
        :param f, shape=(1x1), flow incoming into the node
        :return: shape=(1x1), coalition function output (0 if source node)
        """
        if node_i != 0 or node_i != self.num_tasks:
            coalition_function = getattr(self, self.coalition_types[node_i])
            return coalition_function(f, param=self.coalition_params[node_i])
        else:
            # source and sink node has 0 coalition/reward
            return 0

    def cvar_cost(self, mean, var):
        """

        :param mean:
        :param var:
        :return:
        """
        if not hasattr(self, 'cvar_coeff'):
            #TODO should this be negative? should this look at the lower tail of the distribution instead of the upper??
            alpha = 0.05
            inv_cdf = norm.ppf(alpha)
            numerator = norm.pdf(inv_cdf)
            self.cvar_coeff = numerator/(1-alpha)

        return mean + np.sqrt(var)*self.cvar_coeff  # in the future, change this to a function of the mean and the variance

    def compute_node_reward_dist(self, node_i, node_coalition, reward_mean, reward_variance):
        """
        For a given node, this function outputs the mean and variance of the reward based on the coalition function
        of the node, the reward means of influencing nodes, and the corresponding task influence functions
        TODO: use reward variance of previous nodes in computation as well?
        :param node_i: shape=(1x1), index of node for which coalition function has to be evaluated
        :param node_coalition: shape=(1x1),  coalition value for the node
        :param reward_mean: shape=(num_tasks x 1), mean of rewards for all tasks (only partially filled)
        :param reward_variance: shape=(num_tasks x 1), variance of rewards for all tasks (only partially filled)
        :return: (mean, variance) - two scalars
        """
        # compute incoming edges to node_i
        incoming_edges = list(self.task_graph.in_edges(node_i))

        task_influence_value = []
        for edge in incoming_edges:
            # find global index of edge
            edge_id = list(self.task_graph.edges).index(edge)
            # find the source node of that edge
            source_node = self.edges[edge_id][0]
            # extract the delta function applicable to this edge
            task_interdep = getattr(self, self.dependency_types[edge_id])
            # compute the task influence value (delta for an edge). if "null" then
            if task_interdep.__name__ != 'null':
                task_influence_value.append(task_interdep(reward_mean[source_node],
                                                          self.dependency_params[edge_id]))

        # if reward distribution is not yet initialized, initialize it
        if self.reward_distributions[node_i] is None:
            # rho*delta gives the mean of our reward distribution, where delta is the
            # scalar aggregation of incoming influence func results
            reward_func = lambda rho, delta: rho * delta  # in the future can move this inside reward oracle
            mean_func = lambda reward: reward
            var_func = lambda reward: 0.2 * reward
            influence_agg_func_type = self.influence_agg_func_types[node_i]
            self.reward_distributions[node_i] = RewardOracle(mean_func,
                                                             var_func,
                                                             reward_func,
                                                             influence_agg_func_type, node_id=node_i)

        mean, var = self.reward_distributions[node_i].get_mean_var(node_coalition, task_influence_value)

        return mean, var

    def compute_incoming_flow(self, f):
        """
        Computes the total incoming flow into each node
        :return:
        """
        D_incoming = np.maximum(self.incidence_mat, 0)
        # total incoming flow into each node
        return D_incoming @ f

    def flow_cost(self, f):
        """
        Computes the cost function value over the entire task graph
        :return:
        """
        return np.sum(self.nodewise_optim_cost_function(f))

    def nodewise_optim_cost_function(self, f):
        """
        Computes the cost function value for all the nodes individually based on the flow value
        :param f: shape=(num_edges X 1) , flow value over all edges

        :return: shape=(num_tasks X 1) , cost for each node
        """
        # total incoming flow into each node
        incoming_flow = self.compute_incoming_flow(f)

        var_reward_mean = np.zeros(self.num_tasks, dtype=object)
        var_reward_variance = np.zeros(self.num_tasks, dtype=object)

        node_cost_val = np.zeros(self.num_tasks, dtype=object)

        for node_i in range(self.num_tasks):
            # Compute Coalition Function
            node_coalition = self.compute_node_coalition(node_i, incoming_flow[node_i])
            # Compute the reward by combining with Inter-Task Dependency Function
            # influencing nodes of node i
            var_reward_mean[node_i], var_reward_variance[node_i] = self.compute_node_reward_dist(node_i,
                                                                                                 node_coalition,
                                                                                                 var_reward_mean,
                                                                                                 var_reward_variance)
            # use the cvar metric to compute the cost
            node_cost_val[node_i] = self.cvar_cost(var_reward_mean[node_i],
                                                   var_reward_variance[node_i])

        # return task-wise cost (used in optimization)
        return -node_cost_val

    def update_reward_curves(self):
        """
        Simulates the "disturbance" by changing the reward curves directly
        :return:
        """
        # let's degrade task 2 first
        if self.coalition_params[2][0] > 0.9:
            self.delta = -0.05
        if self.coalition_params[2][0] < 0.1:
            self.delta = 0.05

        self.coalition_params[2][0] = self.coalition_params[2][0] + self.delta

    def initializeSolver(self):
        '''
        This function will define variables, functions, and bounds based on the input info
        :return:
        '''

        self.prog = MathematicalProgram()
        self.var_flow = self.prog.NewContinuousVariables(self.num_edges)

        # Define the Opt program
        # Constraint1: Flow must be positive on all edges
        self.prog.AddConstraint(self.identity,
                                lb=np.zeros(self.num_edges),
                                ub=np.inf * np.ones(self.num_edges),
                                vars=self.var_flow)

        # Constraint2: The inflow must be equal to outflow at all edges
        # compute incidence matrix
        self.incidence_mat = nx.linalg.graphmatrix.incidence_matrix(self.task_graph, oriented=True).A
        b = np.zeros(self.num_tasks)
        b[0] = -self.num_robots
        b[-1] = self.num_robots
        self.prog.AddLinearEqualityConstraint(self.incidence_mat, b, self.var_flow)

        # now for the cost
        self.prog.AddCost(self.flow_cost, vars=self.var_flow)

    def solveGraph(self):
        """
        Solves the optimization program and computes the flow
        :return:
        """
        result = Solve(self.prog)
        print("Success? ", result.is_success())

        print('optimal cost = ', result.get_optimal_cost())
        print('solver is: ', result.get_solver_id().name())
        # Compute coalition values,
        self.flow = result.GetSolution(self.var_flow)
        print('f* = ', self.flow)

    def simulate_task_execution(self):
        """
        Simulate task execution (i.e., sample rewards from distribution) based on already computed flows
        :return:
        """

        incoming_flow = self.compute_incoming_flow(self.flow)
        for node_i in range(self.num_tasks):
            # compute coalition for node_i
            node_coalition = self.compute_node_coalition(node_i, incoming_flow[node_i])
            self.reward_mean[node_i], self.reward_variance[node_i] = self.compute_node_reward_dist(node_i,
                                                                                              node_coalition,
                                                                                              self.reward_mean,
                                                                                              self.reward_variance)
            # sample reward from distribution
            self.reward[node_i] = np.random.normal(self.reward_mean[node_i], self.reward_variance[node_i])

    def sigmoid(self, flow, param):
        return param[0] / (1 + np.exp(-1 * param[1] * (flow - param[2])))

    def dim_return(self, flow, param):
        return param[0] + (param[2] * (1 - np.exp(-1 * param[1] * flow)))

    def null(self, flow, param):
        """
        :param flow:
        :return:
        """
        return 0.0

    def render(self):
        """

        :return:
        """
        if self.fig is None:
            SMALL_SIZE = 10
            MEDIUM_SIZE = 15
            BIGGER_SIZE = 25
            #### FONT SIZE ###########################################################
            plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
            plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
            plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
            plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
            plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
            plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

            # FONT #################################
            plt.rcParams["font.family"] = "Times New Roman"
            plt.rcParams["text.usetex"] = True
            plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

            plt.ion()

            # Aesthetic parameters.
            # Figure aspect ratio.
            fig_aspect_ratio = 16.0 / 9.0  # Aspect ratio of video.
            fig_pixel_height = 1080  # Height of video in pixels.
            dpi = 300  # Pixels per inch (affects fonts and apparent size of inch-scale objects).

            # Set the figure to obtain aspect ratio and pixel size.
            fig_w = fig_pixel_height / dpi * fig_aspect_ratio  # inches
            fig_h = fig_pixel_height / dpi  # inches
            self.fig, self.ax = plt.subplots(1, 1,
                                             figsize=(fig_w, fig_h),
                                             constrained_layout=True,
                                             dpi=dpi)
            # self.ax.set_xlabel('x')
            # self.ax.set_ylabel('y')

            # Setting axis equal should be redundant given figure size and limits,
            # but gives a somewhat better interactive resizing behavior.
            self.ax.set_aspect('equal')

            self.graph_plot_pos = {0: np.array([0, 0.]),
                                   1: np.array([1.0, 0.0]),
                                   2: np.array([1.5, 1.0]),
                                   3: np.array([1.5, -1.0]),
                                   4: np.array([2.0, 0.0])}

            self.graph_plt_handle = nx.drawing.nx_pylab.draw_networkx(self.task_graph,
                                                                      self.graph_plot_pos,
                                                                      arrows=True,
                                                                      with_labels=True,
                                                                      node_color='y',
                                                                      edge_color=self.flow,
                                                                      width=10.0,
                                                                      edge_cmap=plt.cm.Blues,
                                                                      ax=self.ax)
            self.fig.canvas.draw()
            plt.show(block=False)
        else:
            label_dict = {i: format(self.coalition_params[i][0], ".2f") for i in range(self.num_tasks)}
            self.ax.clear()
            self.graph_plt_handle = nx.drawing.nx_pylab.draw_networkx(self.task_graph,
                                                                      self.graph_plot_pos,
                                                                      arrows=True,
                                                                      with_labels=True,
                                                                      labels=label_dict,
                                                                      node_color='y',
                                                                      edge_color=self.flow,
                                                                      width=10.0, edge_cmap=plt.cm.Blues)
            self.fig.canvas.draw()
            plt.show(block=False)
    # def mult(vars):
    #     return np.prod(vars)
    #
    # def add(vars):
    #     return np.sum(vars)
    #
    # def combo(vars):
    #     return np.prod(vars) * np.sum(vars)
    #
    #     # edges_tot will include all edges and their reverses
    #     self.edges_tot = self.edges.copy()
    #
    #     for e in self.edges:
    #         self.edges_tot.append([e[1], e[0]])
    #
    #     self.edge_dict_n = {}
    #     self.edge_dict_r = {}
    #     self.edge_dict_ndex = {}
    #     self.edge_dict_rdex = {}
    #
    #     for i in range(len(self.edges)):
    #         e = self.edges[i]
    #         if e[0] in self.edge_dict_n:
    #             self.edge_dict_n[e[0]].append(e[1])
    #         else:
    #             self.edge_dict_n[e[0]] = [e[1]]
    #
    #         if e[0] in self.edge_dict_ndex:
    #             self.edge_dict_ndex[e[0]].append(i)
    #         else:
    #             self.edge_dict_ndex[e[0]] = [i]
    #
    #         if e[1] in self.edge_dict_r:
    #             self.edge_dict_r[e[1]].append(e[0])
    #         else:
    #             self.edge_dict_r[e[1]] = [e[0]]
    #
    #         if e[1] in self.edge_dict_rdex:
    #             self.edge_dict_rdex[e[1]].append(i)
    #         else:
    #             self.edge_dict_rdex[e[1]] = [i]
    #
    #     for i in range(self.numnodes):
    #         if i not in self.edge_dict_n:
    #             self.edge_dict_n[i] = []
    #         if i not in self.edge_dict_r:
    #             self.edge_dict_r[i] = []
    #         if i not in self.edge_dict_ndex:
    #             self.edge_dict_ndex[i] = []
    #         if i not in self.edge_dict_rdex:
    #             self.edge_dict_rdex[i] = []
    #
    #     print(self.edge_dict_ndex)
    #     print(self.edge_dict_rdex)
    #
    #     # f is the flow across each edge
    #     self.f = self.prog.NewContinuousVariables(len(self.edges_tot), "f")
    #     # p is the coalition component of the reward function for each node
    #     self.p = self.prog.NewContinuousVariables(self.numnodes, "p")
    #     # d is the previous reward component of the reward function for each edge
    #     self.d = self.prog.NewContinuousVariables(len(self.edges), "d")
    #     # r is the reward for each node
    #     self.r = self.prog.NewContinuousVariables(self.numnodes, "r")
    #     # c is the combined flow coming into each node
    #     self.c = self.prog.NewContinuousVariables(self.numnodes, "c")
    #     # g is the aggregation of the deltas coming into each node
    #     self.g = self.prog.NewContinuousVariables(self.numnodes, "g")
    #
    #     # all these variables must be positive
    #     for i in range(self.numnodes):
    #         self.prog.AddConstraint(self.g[i] >= 0)
    #         self.prog.AddConstraint(self.c[i] >= 0)
    #         self.prog.AddConstraint(self.r[i] >= 0)
    #         self.prog.AddConstraint(self.p[i] >= 0)
    #
    #     for i in range(len(self.edges)):
    #         self.prog.AddConstraint(self.d[i] >= 0)
    #
    #     for i in range(len(self.edges)):
    #         # flow cannot exceed number of robots
    #         self.prog.AddConstraint(self.f[i] <= self.numrobots)
    #         # flow over normal edges is inverse of flow on reverse edges
    #         self.prog.AddConstraint(self.f[i] == -1 * self.f[i + len(self.edges)])
    #
    #     for i in range(self.numnodes):
    #         inflow = []
    #         for j in self.edge_dict_rdex[i]:
    #             inflow.append(self.f[j])
    #
    #         inflow = np.array(inflow)
    #
    #         # c[i] is the inflow to node i -- important for rho function
    #         self.prog.AddConstraint(self.c[i] == np.sum(inflow))
    #
    #     # set the inflow of source node to 0
    #     self.prog.AddConstraint(self.c[0] == 0)
    #
    #     for i in range(1, self.numnodes - 1):
    #         outflow = []
    #         for j in self.edge_dict_ndex[i]:
    #             outflow.append(self.f[j])
    #
    #         outflow = np.array(outflow)
    #
    #         # c[i], which is node inflow, must be equal to node outflow (flow conservation)
    #         # this does not apply to the source or the sink
    #         self.prog.AddConstraint(self.c[i] - np.sum(outflow) == 0)
    #
    #     # outflow on node 0 (source) must be equal to number of robots
    #     source_outflow = []
    #     for i in self.edge_dict_ndex[0]:
    #         source_outflow.append(self.f[i])
    #     source_outflow = np.array(source_outflow)
    #     self.prog.AddConstraint(np.sum(source_outflow) == self.numrobots)
    #
    #     # inflow on last node (sink) must be equal to number of robots
    #     self.prog.AddConstraint(self.c[self.numnodes - 1] == self.numrobots)
    #
    #     # reward for source node is just a constant -- 1
    #     self.prog.AddConstraint(self.p[0] == 1)
    #     self.prog.AddConstraint(self.g[0] == 0)
    #     self.prog.AddConstraint(self.r[0] == 1)
    #
    #     # define rho functions as a function of node inflow
    #     for i in range(1, len(self.rhos) + 1):
    #         rho = self.rhos[i - 1]
    #         rhotype = self.rhotypes[i - 1]
    #         if (rhotype == "s"):
    #             print("node", i, ":", *rho)
    #             self.prog.AddConstraint(self.p[i] == TaskGraph.step(rho[0], rho[1], rho[2], self.c[i]))
    #         else:
    #             self.prog.AddConstraint(self.p[i] == TaskGraph.dimin(rho[0], rho[1], rho[2], self.c[i]))
    #
    #     # define delta functions as a function of previous reward
    #     for i in range(len(self.edges)):
    #         delta = self.deltas[i]
    #         deltatype = self.deltatypes[i]
    #         e = self.edges[i]
    #         if (deltatype == "s"):
    #             self.prog.AddConstraint(self.d[i] == TaskGraph.step(delta[0], delta[1], delta[2], self.r[e[0]]))
    #         else:
    #             self.prog.AddConstraint(self.d[i] == TaskGraph.dimin(delta[0], delta[1], delta[2], self.r[e[0]]))
    #
    #     # define agg functions as functions of incoming deltas
    #     for i in range(1, self.numnodes):
    #         agg = self.aggs[i - 1]
    #         indeltas = []
    #         for j in self.edge_dict_rdex[i]:
    #             indeltas.append(self.d[j])
    #
    #         indeltas = np.array(indeltas)
    #
    #         if (agg == "a"):
    #             self.prog.AddConstraint(self.g[i] == TaskGraph.add(indeltas))
    #
    #         elif (agg == "m"):
    #             self.prog.AddConstraint(self.g[i] == TaskGraph.mult(indeltas))
    #
    #         else:
    #             self.prog.AddConstraint(self.g[i] == TaskGraph.combo(indeltas))
    #
    #     # define reward as "combo" of rho and agg
    #     for i in range(1, self.numnodes):
    #         self.prog.AddConstraint(self.r[i] == self.g[i] * self.p[i] * (self.g[i] + self.p[i]))
    #         # here we make the sign of reward negative since the solver is minimizing
    #         # which is equivalent to maximizing the positive reward
    #         self.prog.AddCost(-1 * self.r[i])
