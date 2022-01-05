import numpy as np
from pydrake.solvers.mathematicalprogram import MathematicalProgram
from pydrake.solvers.mathematicalprogram import Solve
#import pydrake.math as math
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
from reward_model import RewardModel
from reward_model_estimate import RewardModelEstimate
from ddp_gym.ddp_gym import DDP
from copy import copy

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

        # someday self.reward_model will hold the ACTUAL values for everything, while self.reward_model_estimate
        # will hold our estimate values
        self.reward_model = RewardModel(num_tasks=self.num_tasks,
                                        num_robots=self.num_robots,
                                        task_graph=self.task_graph,
                                        coalition_params=coalition_params,
                                        coalition_types=coalition_types,
                                        dependency_params=dependency_params,
                                        dependency_types=dependency_types,
                                        influence_agg_func_types=aggs)

        self.reward_model_estimate = RewardModelEstimate(num_tasks=self.num_tasks,
                                num_robots=self.num_robots,
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
        #get current coalition params from reward model estimate
        self.coalition_params = self.reward_model_estimate.get_coalition_params()

        # let's degrade task 2 first
        if self.coalition_params[2][0] > 0.9:
            self.delta = -0.05
        if self.coalition_params[2][0] < 0.1:
            self.delta = 0.05

        # import pdb; pdb.set_trace()
        self.coalition_params[2][0] = self.coalition_params[2][0] + self.delta

        self.reward_modelself.reward_model_estimate.update_coalition_params(self.coalition_params, mode="oracle")

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
        self.prog.AddCost(self.reward_model_estimate.flow_cost, vars=self.var_flow)

    def initialize_solver_ddp(self, constraint_type='qp'):
        # define graph
        num_nodes = 4
        edges = [(0, 1), (1, 2), (2, 3)]
        influence_coeffs = [None, 2, 2, 2]
        coalition_coeffs = [None, 2, 2, 2]

        dynamics_func_handle = self.reward_model.get_dynamics_equations()
        dynamics_func_handle_list = [] #length = num_tasks-1, because no dynamics eqn for first node.
                                       # entry i corresponds to the equation for the reward at node i+1
        cost_func_handle_list = []
        for k in range(1, self.num_tasks):
            dynamics_func_handle_list.append(lambda x, u: dynamics_func_handle(x,u,k))
            cost_func_handle_list.append(lambda x, u: -1*dynamics_func_handle(x,u,k))

        self.ddp = DDP(dynamics_func_handle_list,#[lambda x, u: dynamics_func_handle(x, u, l) for l in range(self.num_tasks)],  # x(i+1) = f(x(i), u)
                  cost_func_handle_list,  # l(x, u) TODO SOMETHING IS GOING ON HERE
                  lambda x: -0.0*x,  # lf(x)
                  100,
                  1,
                  pred_time=self.num_tasks-1,
                  inc_mat=self.reward_model.incidence_mat,
                  adj_mat=self.reward_model.adjacency_mat,
                  edgelist=self.reward_model.edges,
                  constraint_type=constraint_type)
        self.last_u_seq = list(range(self.num_tasks))# = np.ones((self.num_tasks,))
        self.last_x_seq = np.zeros((self.num_tasks,))

        for l in range(0, self.ddp.pred_time):
            incoming_x_seq = self.ddp.x_seq_to_incoming_x_seq(self.last_x_seq)
            incoming_u_seq = self.ddp.u_seq_to_incoming_u_seq(self.last_u_seq)
            breakpoint()
            incoming_rewards_arr = list(incoming_x_seq[l])
            incoming_flow_arr = list(incoming_u_seq[l])
            self.last_x_seq[l+1] = dynamics_func_handle(incoming_rewards_arr, incoming_flow_arr, l + 1)
        print('Initial x_seq: ',self.last_x_seq)
        breakpoint()

    def solve_ddp(self):
        i = 0
        max_iter = 30
        threshold = -1
        delta = np.inf
        prev_u_seq = copy(self.last_u_seq)
        while i < max_iter and delta > threshold:
            k_seq, kk_seq = self.ddp.backward(self.last_x_seq, self.last_u_seq)
            breakpoint()
            self.last_x_seq, self.last_u_seq = self.ddp.forward(self.last_x_seq, self.last_u_seq, k_seq, kk_seq)
            print("states: ",self.last_x_seq)
            print("actions: ",self.last_u_seq)
            i += 1
            delta = np.linalg.norm(np.array(self.last_u_seq) - np.array(prev_u_seq))
            print("iteration ", i-1, " delta: ", delta)
            print("reward: ", -np.sum(self.last_x_seq))
            prev_u_seq = copy(self.last_u_seq)

        self.flow = self.last_u_seq

    def solveGraph(self):
        """
        Solves the optimization program and computes the flow
        :return:
        """
        result = Solve(self.prog)
        print("Success? ", result.is_success())
        breakpoint()
        self.reward_model.flow_cost(result.GetSolution(self.var_flow))
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
        # note that this function uses reward_model - the real-world model of the system - rather than the estimate
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
            edge_labels_dict = {}
            for j in range(self.task_graph.number_of_edges()):
                edge_labels_dict[list(self.task_graph.edges)[j]] = format(self.flow[j], ".2f")
            nx.drawing.nx_pylab.draw_networkx_edge_labels(self.task_graph,
                                                          self.graph_plot_pos,
                                                          edge_labels=edge_labels_dict)


            self.fig.canvas.draw()
            plt.show(block=False)
