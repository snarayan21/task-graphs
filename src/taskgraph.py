import numpy as np
from pydrake.solvers.mathematicalprogram import MathematicalProgram
from pydrake.solvers.mathematicalprogram import Solve
import pydrake.math as math
from scipy.optimize import minimize, LinearConstraint
import cyipopt

import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
from reward_model import RewardModel
from reward_model_estimate import RewardModelEstimate
from ddp_gym.ddp_gym import DDP
from copy import copy

from scipt_minlp import MRTA_XD

from autograd import grad

import os


class TaskGraph:
    # class for task graphs where nodes are tasks and edges are precedence relationships

    def __init__(self, max_steps, num_tasks, edges, coalition_params, coalition_types, dependency_params, dependency_types, aggs,
                 numrobots, task_times=None, scenario="test", adaptive=1):
        self.scenario = scenario
        self.adaptive = adaptive
        self.max_steps = max_steps
        self.num_tasks = num_tasks
        self.num_robots = numrobots
        self.task_graph = nx.DiGraph()
        self.task_graph.add_nodes_from(range(num_tasks))
        self.task_graph.add_edges_from(edges)
        self.num_edges = len(edges)  # number of edges

        if task_times is None:
            task_times = np.random.rand(num_tasks) # randomly sample task times from the range 0 to 1
        self.task_times = task_times

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

        self.minlp_obj = MRTA_XD(
            num_tasks=self.num_tasks,
            num_robots=self.num_robots,
            dependency_edges=edges,
            coalition_params=coalition_params,
            coalition_types=coalition_types,
            dependency_params=dependency_params,
            dependency_types=dependency_types,
            influence_agg_func_types=aggs,
            reward_model=self.reward_model,
            task_graph = self.task_graph,
            task_times = self.task_times
        )

        # variables using in the optimization
        self.var_flow = None

        # variables used during run-time
        self.flow = None
        self.reward = np.zeros(self.num_tasks)

        # variables used for DDP
        self.alpha_anneal = False
        self.constraint_buffer = False

        #variables used for data logging
        self.last_baseline_solution = None
        self.last_ddp_solution = None
        self.last_minlp_solution = None
        self.last_minlp_solution_val = None
        self.ddp_reward_history = None
        self.last_greedy_solution = None
        self.constraint_residual = None
        self.alpha_hist = None
        self.buffer_hist = None
        self.constraint_violation = None



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

        if "test" in self.scenario:
            # let's degrade task 2 first
            if self.coalition_params[2][0] > 0.9:
                self.delta = -0.05
            if self.coalition_params[2][0] < 0.1:
                self.delta = 0.05

            self.coalition_params[2][0] = self.coalition_params[2][0] + self.delta
        elif "farm" in self.scenario:
            ######## TEST 1 (Break the symmetry between 1 and 3) ###############################
            # let's degrade task 2 first


            ######## TEST 2 (Make prep feeding task 7 infeasible) #######################
            if self.coalition_params[7][0] > 0.9:
                self.delta = -0.05
            if self.coalition_params[7][0] < 0.1:
                self.delta = 0.01

            self.coalition_params[7][0] = self.coalition_params[7][0] + self.delta
        #TODO: Implement the adaptive piece here
        self.reward_model_estimate.update_coalition_params(self.coalition_params, mode="oracle")

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

    def solve_graph_minlp(self):
        self.minlp_obj.model.optimize()
        self.last_minlp_solution_val = self.minlp_obj.model.getObjVal()

        xak_list = [self.minlp_obj.model.getVal(self.minlp_obj.x_ak[i]) for i in range(len(self.minlp_obj.x_ak))]
        oakk_list = [self.minlp_obj.model.getVal(self.minlp_obj.o_akk[i]) for i in range(len(self.minlp_obj.o_akk))]
        oakk_np = np.reshape(np.array(oakk_list),(self.num_robots,self.num_tasks+1,self.num_tasks))
        zak_list = [self.minlp_obj.model.getVal(self.minlp_obj.z_ak[i]) for i in range(len(self.minlp_obj.z_ak))]
        sk_list = [self.minlp_obj.model.getVal(self.minlp_obj.s_k[i]) for i in range(len(self.minlp_obj.s_k))]
        fk_list = [self.minlp_obj.model.getVal(self.minlp_obj.f_k[i]) for i in range(len(self.minlp_obj.f_k))]

        verbose = False
        if verbose:
            print("MINLP SOLUTION COMPLETE. OBJECTIVE VALUE: ", self.last_minlp_solution_val)
            print("x_ak:", xak_list)
            print("o_akk:", oakk_np)
            print("z_ak:", zak_list)
            print("s_k:", sk_list)
            print("f_k:", fk_list)


        # reshape o_akk so that o_akk[a, k-1, k'] = 1 --> agent a performs task k' immediately after task k
        # index 0 in dimension 2 is for the dummy tasks. self-edges not included for dummy tasks, but included for all others
        for a in range(self.num_robots):
            for k in range(self.num_tasks):
                for k_p in range(self.num_tasks):
                    if oakk_np[a,k+1,k_p] == 1:
                        if verbose:
                            print("Agent ", a, " performs task ", k, " and then task ", k_p)
        self.last_minlp_solution = np.array(xak_list + oakk_list + zak_list + sk_list + fk_list)
        self.translate_minlp_objective(self.last_minlp_solution)

    def solve_graph_scipy(self):
        self.incidence_mat = nx.linalg.graphmatrix.incidence_matrix(self.task_graph, oriented=True).A
        b = np.zeros(self.num_tasks)

        # scipy version
        # equality flow constraint
        lb2 = np.zeros(self.num_tasks-2)
        ub2 = np.zeros(self.num_tasks-2)
        c2 = LinearConstraint(self.incidence_mat[1:-1,:], lb=lb2, ub=ub2)

        # inequality constraint on edge capacity
        c1 = LinearConstraint(np.eye(self.num_edges),
                               lb = np.zeros(self.num_edges),
                               ub = np.ones(self.num_edges))

        # inequality constraint on beginning and ending flow
        c3 = LinearConstraint(self.incidence_mat[0,:],lb=[-1],ub=0)
        #import pdb; pdb.set_trace()
        scipy_result = minimize(self.reward_model.flow_cost, np.ones(self.num_edges)*0.5, constraints=(c1,c2,c3))
        print(scipy_result)
        self.last_baseline_solution = scipy_result

    def solve_graph_greedy(self):
        initial_flow = 1.0
        self.last_greedy_solution = np.zeros((self.num_edges,))
        num_assigned_edges = 0
        node_queue = []
        curr_node = 0

        while True:
            out_edges = self.task_graph.out_edges(curr_node)
            out_edge_inds = [list(self.task_graph.edges).index(edge) for edge in out_edges]
            in_edges = list(self.task_graph.in_edges(curr_node))
            in_edge_inds = [list(self.task_graph.edges).index(edge) for edge in in_edges]
            if num_assigned_edges == self.num_edges:
                break
            num_edges = len(out_edges)

            # make cost function handle that takes in edge values and returns rewards
            def node_reward(f, arg_curr_node, arg_num_assigned_edges):
                input_flows = np.concatenate((self.last_greedy_solution[0:arg_num_assigned_edges], f, np.zeros((self.num_edges-len(f)-arg_num_assigned_edges,))))
                rewards = -1*self.reward_model._nodewise_optim_cost_function(input_flows)
                relevant_reward_inds = list(range(arg_curr_node+1))
                for n in self.task_graph.neighbors(arg_curr_node):
                    relevant_reward_inds.append(n)

                relevant_costs = rewards[relevant_reward_inds]
                return np.sum(relevant_costs)

            # get incoming flow quantity to node
            incoming_flow = np.sum(self.last_greedy_solution[in_edge_inds])
            if curr_node == 0:
                incoming_flow = 1.0
            #node_cost(0.5*np.ones((num_edges,)), curr_node, num_assigned_edges)

            # use random sampling to find a good initial state
            candidate_flows = []
            cand_flow_rewards = []
            n_samples = 50
            for n in range(n_samples):
                cand_flow = np.random.rand(num_edges)
                cand_flow = incoming_flow*cand_flow/np.sum(cand_flow)
                candidate_flows.append(cand_flow)
                cand_flow_rewards.append(node_reward(cand_flow,curr_node,num_assigned_edges))

            #find best initial state NOTE: finding max reward
            best_ind = np.argmax(np.array(cand_flow_rewards))
            best_init_state = candidate_flows[best_ind]
            gradient_func = grad(node_reward,0)

            # GRADIENT DESCENT
            max_iter = 50
            dt = 0.1
            last_state = best_init_state
            for i in range(max_iter):
                # take gradient of cost function with respect to edge values
                gradient_t = gradient_func(last_state,curr_node,num_assigned_edges)

                # TODO: project gradient onto hyperplane that respects constraints
                # FOR NOW: just normalize new state such that it is valid

                # take a step along that vector direction
                new_cand_state = last_state + dt*gradient_t
                for k in range(num_edges):
                    if new_cand_state[k] < 0:
                        new_cand_state[k] = 0.00001
                last_state = incoming_flow*new_cand_state/np.sum(new_cand_state)


            # update self.last_greedy_solution
            for (edge_i, new_flow) in zip(out_edge_inds,last_state):
                self.last_greedy_solution[edge_i] = new_flow
            # continue to next node
            out_nbrs = [n for n in self.task_graph.neighbors(curr_node)]
            node_queue.extend(out_nbrs)
            curr_node = node_queue.pop(0)
            num_assigned_edges += num_edges


    def initialize_solver_ddp(self, constraint_type='qp', constraint_buffer='soft', alpha_anneal='True', flow_lookahead='False'):
        self.alpha_anneal = alpha_anneal
        self.constraint_buffer = constraint_buffer

        dynamics_func_handle = self.reward_model.get_dynamics_equations()
        dynamics_func_handle_list = [] #length = num_tasks-1, because no dynamics eqn for first node.
                                       # entry i corresponds to the equation for the reward at node i+1
        cost_func_handle_list = []
        for k in range(1, self.num_tasks):
            dynamics_func_handle_list.append(lambda x, u, additional_x, l_index, k=k: dynamics_func_handle(x,u,k,additional_x,l_index))
            cost_func_handle_list.append(lambda x, u, additional_x, l_index, k=k: -1*dynamics_func_handle(x,u,k,additional_x,l_index))

        self.ddp = DDP(dynamics_func_handle_list,#[lambda x, u: dynamics_func_handle(x, u, l) for l in range(self.num_tasks)],  # x(i+1) = f(x(i), u)
                  cost_func_handle_list,  # l(x, u)
                  lambda x: -0.0*x,  # lf(x)
                  100,
                  1,
                  pred_time=self.num_tasks - 1,
                  inc_mat=self.reward_model.incidence_mat,
                  adj_mat=self.reward_model.adjacency_mat,
                  edgelist=self.reward_model.edges,
                  constraint_type=constraint_type,
                  constraint_buffer=constraint_buffer,
                  flow_lookahead=flow_lookahead)

        self.last_u_seq = np.zeros((self.num_edges,))#list(range(self.num_edges))
        self.last_x_seq = np.zeros((self.num_tasks,))

        incoming_nodes = self.ddp.get_incoming_node_list()
        for l in range(0, self.ddp.pred_time):
            incoming_x_seq = self.ddp.x_seq_to_incoming_x_seq(self.last_x_seq)
            incoming_u_seq = self.ddp.u_seq_to_incoming_u_seq(self.last_u_seq)
            incoming_rewards_arr = list(incoming_x_seq[l])
            incoming_flow_arr = list(incoming_u_seq[l])
            if l in incoming_nodes[l]:
                l_ind = incoming_nodes[l].index(l)
                x = incoming_rewards_arr[l_ind]
                incoming_rewards_arr.pop(l_ind)
                additional_x = incoming_rewards_arr
            else:
                l_ind = -1
                additional_x = incoming_rewards_arr
                x = None
            #breakpoint()

            self.last_x_seq[l+1] = dynamics_func_handle(x, incoming_flow_arr, l + 1, additional_x,l_ind)
        print('Initial x_seq: ',self.last_x_seq)

    def solve_ddp(self):
        i = 0
        max_iter = 100
        buffer = 0.1
        alpha = 0.5
        threshold = -1
        delta = np.inf
        prev_u_seq = copy(self.last_u_seq)
        reward_history = []
        constraint_residual = []
        constraint_violations = []
        alpha_hist = []
        buffer_hist = []


        while i < max_iter and delta > threshold:
            #print("new iteration!!!!")
            #breakpoint()
            if(self.constraint_buffer == 'True'):
                buf = buffer - ((buffer * i)/(max_iter-1))
            else:
                buf = 0

            if(self.alpha_anneal == 'True'):
                curr_alpha = (alpha/(i+1)**(1/3))

            k_seq, kk_seq = self.ddp.backward(self.last_x_seq, self.last_u_seq, max_iter, i, buf, curr_alpha)
            #breakpoint()
            #np.set_printoptions(suppress=True)

            print("alpha is: ", curr_alpha)
            self.last_x_seq, self.last_u_seq = self.ddp.forward(self.last_x_seq, self.last_u_seq, k_seq, kk_seq, i, curr_alpha)
            print("states: ",self.last_x_seq)
            print("actions: ", self.last_u_seq)
            i += 1
            delta = np.linalg.norm(np.array(self.last_u_seq) - np.array(prev_u_seq))
            print("iteration ", i-1, " delta: ", delta)
            print("reward: ", np.sum(self.last_x_seq))

            # log data
            reward_history.append(np.sum(self.last_x_seq))
            constraint_residual.append(self.get_constraint_residual(self.last_u_seq))
            alpha_hist.append(curr_alpha)
            buffer_hist.append(buf)
            prev_u_seq = copy(self.last_u_seq)
            
            #compute constraint violations from last_u_seq
            inc_mat = self.reward_model.incidence_mat
            total_violation = 0.0
            for l in range(self.num_tasks):
                curr_inc_mat = inc_mat[l]
                #curr node inflow
                u = 0.0
                for j in range(len(curr_inc_mat)):
                    if(curr_inc_mat[j] == 1):
                        u += self.last_u_seq[j]
                #curr node outflow
                p = 0.0
                for j in range(len(curr_inc_mat)):
                    if(curr_inc_mat[j] == -1):
                        p += self.last_u_seq[j]
                
                if(u < 0.0):
                    total_violation += 0-u
                if(p < 0.0):
                    total_violation += 0-p
                if(u > 1.0):
                    total_violation += u-1
                if(p > 1.0):
                    total_violation += p-1

            constraint_violations.append(total_violation)
            print("total constraint violation is: ", total_violation)

        self.flow = self.last_u_seq
        self.last_ddp_solution = self.last_u_seq
        self.ddp_reward_history = reward_history
        self.constraint_residual = constraint_residual
        self.alpha_hist = alpha_hist
        self.buffer_hist = buffer_hist
        self.constraint_violation = constraint_violations

    def solveGraph(self):
        result = Solve(self.prog)
        print("Success? ", result.is_success())
        #breakpoint()
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

    def time_task_execution(self, flow):
        frontier_nodes = []
        task_start_times = np.zeros((self.num_tasks,))
        task_finish_times = np.zeros((self.num_tasks,))

        nodelist = list(range(self.num_tasks))
        frontier_nodes.append(nodelist[0])
        while len(frontier_nodes) > 0:
            current_node = frontier_nodes.pop(0)
            incoming_edges = [list(e) for e in self.task_graph.in_edges(current_node)]
            #print(current_node)
            #print(frontier_nodes)
            #print(incoming_edges)
            #incoming_nodes = [n for n in self.task_graph.predecessors(current_node)]
            if len(incoming_edges) > 0:
                #incoming_edge_inds = [self.reward_model.edges.index(e) for e in incoming_edges]
                if np.array([flow[e[0]]<=0.000001 for e in incoming_edges]).all():
                    task_start_times[current_node] = 0.0
                else:
                    task_start_times[current_node] = max([task_finish_times[e[0]] for e in incoming_edges if (flow[e[0]]>0.000001)])
            else:
                task_start_times[current_node] = 0
            task_finish_times[current_node] = task_start_times[current_node] + self.task_times[current_node]
            for n in self.task_graph.neighbors(current_node):
                frontier_nodes.append(n)

        return task_start_times, task_finish_times




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

    def discretize(self, num_edges):
        candidate_points = []
        #for edge_i in range(num_edges):
        # JUST RANDOM SAMPLE FOR NOW BC I DON'T WANT TO WASTE MORE TIME ON THIS

        return candidate_points

    def get_constraint_residual(self, u_seq):
        incoming_u_seq = self.ddp.u_seq_to_incoming_u_seq(u_seq) # NOTE DIFFERENT INDEXING -- index i corresponds to inflow to node i+1
        outgoing_u_seq = self.ddp.u_seq_to_outgoing_u_seq(u_seq) # NOTE DIFFERENT INDEXING -- index i corresponds to outflow from node i
        residuals = []
        for i in range(1,len(incoming_u_seq)):
            incoming_f = np.sum(incoming_u_seq[i-1])
            outgoing_f = np.sum(outgoing_u_seq[i])
            residuals.append(outgoing_f-incoming_f)
        return np.linalg.norm(np.array(residuals))

    def test_minlp(self):

        case_a = False
        case_b = True
        if case_a:
            x_ak = np.zeros(((self.num_tasks+1)*self.num_robots,))
            o_akk = np.zeros((self.num_robots,self.num_tasks+1, self.num_tasks),)
            z_ak = np.zeros(((self.num_tasks+1)*self.num_robots,))
            s_k = 2*np.arange(self.num_tasks)
            f_k = s_k+1

            x_ak[0] = 1 # dummy task
            x_ak[1] = 1
            x_ak[2] = 1
            x_ak[3] = 1
            x_ak[5] = 1

            o_akk[0,0,0] = 1 # task __ -> 0
            o_akk[0,1,1] = 1 # task 0 -> 1
            o_akk[0,2,2] = 1 # task 1 -> 2
            o_akk[0,3,4] = 1 # task 2 -> 4

            print(x_ak, o_akk, z_ak, s_k, f_k)

            #x_vec = np.concatenate((x_ak, o_akk.flatten(), z_ak,s_k,f_k))
            #self.minlp_obj.objective(x_vec)


        if case_b:
            x_ak = np.zeros(((self.num_tasks+1)*self.num_robots,))
            o_akk = np.zeros((self.num_robots,self.num_tasks+1, self.num_tasks),)
            z_ak = np.zeros(((self.num_tasks+1)*self.num_robots,))
            s_k = 2*np.arange(self.num_tasks)
            f_k = s_k+1

            #agent 0 does tasks 0, 1, 2, 4
            x_ak[0] = 1 # dummy task
            x_ak[1] = 1
            x_ak[2] = 1
            x_ak[3] = 1
            x_ak[5] = 1


            o_akk[0,0,0] = 1 # task __ -> 0
            o_akk[0,1,1] = 1 # task 0 -> 1
            o_akk[0,2,2] = 1 # task 1 -> 2
            o_akk[0,3,4] = 1 # task 2 -> 4

            #agent 1 does tasks 0, 1, 3, 4
            x_ak[6] = 1 # dummy task
            x_ak[7] = 1
            x_ak[8] = 1
            x_ak[10] = 1
            x_ak[11] = 1

            o_akk[1,0,0] = 1 # task __ -> 0
            o_akk[1,1,1] = 1 # task 0 -> 1
            o_akk[1,2,3] = 1 # task 1 -> 3
            o_akk[1,4,4] = 1 # task 3 -> 4

            print(x_ak,o_akk,z_ak,s_k,f_k)
            #x_vec = np.concatenate((x_ak, o_akk.flatten(), z_ak,s_k,f_k))
            #self.minlp_obj.objective(x_vec)

        cons = self.minlp_obj.model.getConss()
        for c in cons:
            print(c)
        breakpoint()
        self.minlp_obj.model.optimize()
        print(self.minlp_obj.model.getObjVal())
        xak_list = [self.minlp_obj.model.getVal(self.minlp_obj.x_ak[i]) for i in range(len(self.minlp_obj.x_ak))]
        print("x_ak:", xak_list)
        oakk_list = [self.minlp_obj.model.getVal(self.minlp_obj.o_akk[i]) for i in range(len(self.minlp_obj.o_akk))]
        oakk_np = np.reshape(np.array(oakk_list),(self.num_robots,self.num_tasks+1,self.num_tasks))
        print("o_akk:", oakk_np)
        zak_list = [self.minlp_obj.model.getVal(self.minlp_obj.z_ak[i]) for i in range(len(self.minlp_obj.z_ak))]
        print("z_ak:", zak_list)
        sk_list = [self.minlp_obj.model.getVal(self.minlp_obj.s_k[i]) for i in range(len(self.minlp_obj.s_k))]
        print("s_k:", sk_list)
        fk_list = [self.minlp_obj.model.getVal(self.minlp_obj.f_k[i]) for i in range(len(self.minlp_obj.f_k))]
        print("f_k:", fk_list)
        # reshape o_akk so that o_akk[a, k-1, k'] = 1 --> agent a performs task k' immediately after task k
        # index 0 in dimension 2 is for the dummy tasks. self-edges not included for dummy tasks, but included for all others
        for a in range(self.num_robots):
            for k in range(self.num_tasks):
                for k_p in range(self.num_tasks):
                    if oakk_np[a,k+1,k_p] == 1:
                        print("Agent ", a, " performs task ", k, " and then task ", k_p)
        minlp_objective = np.array(xak_list + oakk_list + zak_list + sk_list + fk_list)
        self.translate_minlp_objective(minlp_objective)
        breakpoint()
        import pdb; pdb.set_trace()

    def translate_minlp_objective(self, x):
        x_ak, o_akk, z_ak, s_k, f_k = self.minlp_obj.partition_x(x)
        # x_ak organized by agent
        x_ak = np.reshape(np.array(x_ak), (self.num_robots, self.num_tasks + 1)) # reshape so each row contains x_ak for agent a
        x_dummy = x_ak[:,0]
        x_ak = x_ak[:,1:]
        task_coalitions = []#np.zeros((self.num_tasks,)) # list of coalition size assigned to each task
        for t in range(self.num_tasks):
            task_coalitions.append(np.sum(x_ak[:,t]))
        print("task coalitions: ",task_coalitions)
        o_akk = np.atleast_3d(np.reshape(o_akk,(self.num_robots,self.num_tasks+1, self.num_tasks)))
        for a in range(self.num_robots):
            for j in range(self.num_tasks):
                for k in range(self.num_tasks):
                    if(o_akk[a,j+1,k]>0.99):
                        print("agent %d performs task %d before task %d" %(a,j,k))

        tasks_ordered = np.argsort(np.array(f_k))
        for t in tasks_ordered:
            print("Time ", f_k[t],": task %d completed by %d agents" % (t, task_coalitions[t]))


def discretize_pairwise(max_val):
    """ Creates a list of pairs of flows. Each pair sums to max_val, and it is discretized by an interval of 0.1"""
    flow_a = np.arange(start=0, stop=max_val+0.1, step=0.1)
    flow_b = max_val-flow_a
    import pdb; pdb.set_trace()
    return np.concatenate((np.expand_dims(flow_a,1),np.expand_dims(flow_b,1)),axis=1).tolist()
