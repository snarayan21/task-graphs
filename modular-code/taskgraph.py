import numpy as np
from pydrake.solvers.mathematicalprogram import MathematicalProgram
from pydrake.solvers.mathematicalprogram import Solve
import pydrake.math as math
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
from reward_model import RewardModel
import os


class TaskGraph:
    # class for task graphs where nodes are tasks and edges are precedence relationships

    def __init__(self, num_tasks, edges, coalition_params, coalition_types, dependency_params, dependency_types, aggs,
                 numrobots):
        self.num_tasks = num_tasks
        self.num_robots = numrobots
        self.task_graph = nx.DiGraph()
        self.task_graph.add_nodes_from(range(num_tasks))
        self.task_graph.add_edges_from(edges)
        self.num_edges = len(edges)  # number of edges
        self.fig = None

        self.reward_model = RewardModel(num_tasks=self.num_tasks,
                                        num_robots=self.num_robots,
                                        edges=edges,
                                        task_graph=self.task_graph,
                                        coalition_params=coalition_params,
                                        coalition_types=coalition_types,
                                        dependency_params=dependency_params,
                                        dependency_types=dependency_types,
                                        influence_agg_func_types=aggs)

        # variables using in the optimization
        self.var_flow = None

        # variables used during run-time
        self.flow = None
        self.reward = np.zeros(self.num_tasks)



    def identity(self, f):
        """
        Identity function (for passing into pydrake)
        :return:
        """
        return f



    def update_reward_curves(self):
        """
        Simulates the "disturbance" by changing the reward curves directly
        :return:
        """
        self.reward_model.update_coalition_params()

    def initializeSolver(self):
        '''
        This function will define variables, functions, and bounds based on the input info
        :return:
        '''

        self.prog = MathematicalProgram()
        self.var_flow = self.prog.NewContinuousVariables(self.num_edges)

        # Define the Opt program
        # Constraint1: Flow must be positive on all edges and can never exceed 1
        self.prog.AddConstraint(self.identity,
                                lb=np.zeros(self.num_edges),
                                ub=np.ones(self.num_edges),
                                vars=self.var_flow)

        # Constraint2: The inflow must be equal to outflow at all edges
        # compute incidence matrix
        self.incidence_mat = nx.linalg.graphmatrix.incidence_matrix(self.task_graph, oriented=True).A
        b = np.zeros(self.num_tasks)
        b[0] = -1.0#self.num_robots # flow constrained to sum upto 1
        b[-1] = 1.0#self.num_robots
        self.prog.AddLinearEqualityConstraint(self.incidence_mat, b, self.var_flow)

        # now for the cost
        self.prog.AddCost(self.reward_model.flow_cost, vars=self.var_flow)

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

        self.reward = self.reward_model._nodewise_optim_cost_function(self.flow, eval=True)



    def render(self):
        """

        :return:
        """
        matplotlib.use('TKAgg', force=True)
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
            if self.num_tasks == 5:
                self.graph_plot_pos = {0: np.array([0, 0.]),
                                       1: np.array([1.0, 0.0]),
                                       2: np.array([1.5, 1.0]),
                                       3: np.array([1.5, -1.0]),
                                       4: np.array([2.0, 0.0])}
            else:
                self.graph_plot_pos = nx.planar_layout(self.task_graph)

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
            label_dict = {i: format(self.reward_model.coalition_params[i][0], ".2f") for i in range(self.num_tasks)}
            self.ax.clear()
            self.graph_plt_handle = nx.drawing.nx_pylab.draw_networkx(self.task_graph,
                                                                      self.graph_plot_pos,
                                                                      arrows=True,
                                                                      with_labels=True,
                                                                      labels=label_dict,
                                                                      node_color='y',
                                                                      edge_color=self.flow,
                                                                      width=10.0, edge_cmap=plt.cm.Blues)
            import pdb; pdb.set_trace()
            edge_labels_dict = {}
            for j in range(self.task_graph.number_of_edges()):
                edge_labels_dict[list(self.task_graph.edges)[j]] = format(self.flow[j], ".2f")
            nx.drawing.nx_pylab.draw_networkx_edge_labels(self.task_graph,
                                                          self.graph_plot_pos,
                                                          edge_labels=edge_labels_dict)


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
