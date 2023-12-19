import numpy as np
from scipy.optimize import minimize, LinearConstraint

import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
from reward_model import RewardModel
from copy import copy

from scipt_minlp import MRTA_XD

from autograd import grad

import pdb
import os

class TaskGraph:
    # class for task graphs where nodes are tasks and edges are precedence relationships

    def __init__(self,
                 num_tasks,
                 edges,
                 coalition_params,
                 coalition_types,
                 dependency_params,
                 dependency_types,
                 aggs,
                 num_robots,
                 nodewise_coalition_influence_agg_list=None,
                 task_times=None,
                 makespan_constraint=10000,
                 minlp_time_constraint=False, # set to False for no time constraint, set to integer number of seconds for a time constraint
                 minlp_reward_constraint=False,
                 run_minlp=True,
                 coalition_influence_aggregator=None,
                 warm_start=False,
                 npl=None):

        self.num_tasks = num_tasks
        self.num_robots = num_robots
        self.nodewise_coalition_influence_agg_list = nodewise_coalition_influence_agg_list
        if not minlp_time_constraint:
            self.minlp_time_constraint = 10000000000
        else:
            self.minlp_time_constraint = minlp_time_constraint
        self.minlp_reward_constraint = minlp_reward_constraint
        self.task_graph = nx.DiGraph()
        self.task_graph.add_nodes_from(range(num_tasks))
        self.task_graph.add_edges_from(edges)
        self.num_edges = len(edges)  # number of edges

        self.edges = [edge for edge in self.task_graph.edges]
        self.coalition_params = coalition_params
        self.coalition_types = coalition_types
        self.dependency_params = dependency_params
        self.dependency_types = dependency_types
        self.aggs = aggs

        self.run_minlp = run_minlp
        self.warm_start = warm_start
        self.coalition_influence_aggregator = coalition_influence_aggregator

        if task_times is None:
            task_times = np.random.rand(num_tasks)# randomly sample task times from the range 0 to 1
            task_times[0] = 0.0

        self.task_times = np.array(task_times, dtype='float')
        print("self.task_times: ", self.task_times)

        # makespan constriant is 'cleared' if we have already pruned the graph and are just using this object to
        # calculate pruned solutions
        # makespan constraint represents RELATIVE VALUE of total task duration (between 0 and 1) otherwise
        if makespan_constraint == 'cleared':
            self.makespan_constraint = makespan_constraint
        else:
            self.makespan_constraint = np.sum(self.task_times)*float(makespan_constraint)
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
                                        influence_agg_func_types=aggs,
                                        nodewise_coalition_influence_agg_list=self.nodewise_coalition_influence_agg_list)

        self.pruned_graph_list = None
        self.pruned_graph_edge_mappings_list = None

        if self.makespan_constraint != 'cleared':
            print("not cleared")
            if self.run_minlp:
                self.minlp_obj = self.initialize_minlp_obj()
            print("pruning graph...")
            self.pruned_graph_list, self.pruned_graph_edge_mappings_list = self.prune_graph()

        # variables used during run-time
        self.flow = None
        self.reward = np.zeros(self.num_tasks)


        #variables used for data logging
        self.last_baseline_solution = None
        self.rounded_baseline_solution = None
        self.last_minlp_solution = None
        self.last_minlp_solution_val = None
        self.last_greedy_solution = None
        self.constraint_residual = None
        self.alpha_hist = None
        self.buffer_hist = None
        self.constraint_violation = None

    def initialize_minlp_obj(self):
        obj = MRTA_XD(
            num_tasks=self.num_tasks,
            num_robots=self.num_robots,
            dependency_edges=self.edges,
            coalition_params=self.coalition_params,
            coalition_types=self.coalition_types,
            dependency_params=self.dependency_params,
            dependency_types=self.dependency_types,
            influence_agg_func_types=self.aggs,
            nodewise_coalition_influence_agg_list=self.nodewise_coalition_influence_agg_list,
            reward_model=self.reward_model,
            task_graph=self.task_graph,
            task_times=self.task_times,
            makespan_constraint=self.makespan_constraint
        )
        return obj


    def prune_graph(self):
        print("Pruning Graph...")
        start_times, finish_times = self.time_task_execution(np.ones((self.num_edges,)))
        to_prune = finish_times > self.makespan_constraint
        to_prune_indices = [i for i in range(self.num_tasks) if to_prune[i]]

        # create list of pruned indices
        pruned_tasks = [i for i in range(self.num_tasks) if not to_prune[i]]

        pruned_edges = []         # create list of pruned edges
        pruned_edges_mapping = []         # create mapping from original edges to pruned edges
        for (edge, edge_ind) in zip(self.edges,range(len(self.edges))):
            if int(edge[0]) in pruned_tasks and int(edge[1]) in pruned_tasks:
                edge = [int(i) for i in edge]
                pruned_edges.append(edge)
                pruned_edges_mapping.append(edge_ind)

        # rename edges according to new number of tasks
        renamed_edges = []
        for edge in pruned_edges:
            node_out = pruned_tasks.index(int(edge[0]))
            node_in = pruned_tasks.index(int(edge[1]))
            renamed_edges.append((node_out,node_in))

        # assemble new task graph args
        coalition_types = []
        coalition_params = []
        aggs = []
        for task in pruned_tasks:
            coalition_types.append(self.coalition_types[task])
            coalition_params.append(self.coalition_params[task])
            aggs.append(self.aggs[task])

        dependency_types = []
        dependency_params = []
        for edge_ind in range(len(pruned_edges)):
            dependency_types.append(self.dependency_types[pruned_edges_mapping[edge_ind]])
            dependency_params.append(self.dependency_params[pruned_edges_mapping[edge_ind]])

        # create new task graph
        new_graph = TaskGraph(
                 num_tasks=len(pruned_tasks), # new quantity of tasks
                 edges=renamed_edges, # new edges
                 coalition_params=coalition_params,
                 coalition_types=coalition_types,
                 dependency_params=dependency_params,
                 dependency_types=dependency_types,
                 aggs=aggs,
                 num_robots=self.num_robots,
                 nodewise_coalition_influence_agg_list=self.nodewise_coalition_influence_agg_list,
                 task_times=self.task_times,
                 makespan_constraint='cleared', # specify cleared because it is already pruned
                 minlp_time_constraint=False, # these shouldn't matter -- MINLP will not be initialized
                 minlp_reward_constraint=False
        )

        print("New Graph After Pruning: ", new_graph.task_graph)
        return [new_graph], [pruned_edges_mapping]

    def identity(self, f):
        """
        Identity function (for passing into pydrake)
        :return:
        """
        return f

    def solve_graph_minlp(self):
        self.minlp_obj.model.setParam('limits/time', self.minlp_time_constraint)
        if self.warm_start:
            self.minlp_warm_start(self.pruned_rounded_baseline_solution)
        self.minlp_obj.model.optimize()
        status = self.minlp_obj.model.getStatus()
        try:
            self.last_minlp_solution_val = self.minlp_obj.model.getObjVal()
        except:

            # if status != "optimal" and status != "timelimit":
            x_len = (self.num_tasks+1)*self.num_robots # extra dummy task
            o_len = self.num_robots*((self.num_tasks+1)*(self.num_tasks)) #extra dummy task, KEEP duplicates
            z_len = (self.num_tasks + 1)*self.num_robots #each agent can finish on each task -- include dummy task, as agents can do 0 tasks
            s_len = self.num_tasks
            f_len = self.num_tasks
            self.last_minlp_solution_val = 0.0
            self.last_minlp_solution = np.zeros((x_len+o_len+z_len+s_len+f_len,))
            self.minlp_makespan = 0.0
            self.minlp_dual_bound = self.minlp_obj.model.getDualbound()
            self.minlp_primal_bound = 1000000000
            self.minlp_gap = 1000000000
            self.minlp_obj_limit = 1000000000
            return

        self.minlp_dual_bound = self.minlp_obj.model.getDualbound()
        self.minlp_primal_bound = self.minlp_obj.model.getPrimalbound()
        self.minlp_gap = self.minlp_obj.model.getGap()
        self.minlp_obj_limit = self.minlp_obj.model.getObjlimit()
        print(self.minlp_dual_bound)
        print(self.minlp_primal_bound)
        print(self.minlp_gap)
        print(self.minlp_obj_limit)
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
        self.minlp_info_dict = self.translate_minlp_objective(self.last_minlp_solution)
        self.minlp_makespan = self.minlp_info_dict['makespan']


    def minlp_warm_start(self, flow):
        x_len = (self.num_tasks+1)*self.num_robots # extra dummy task
        o_len = self.num_robots*((self.num_tasks+1)*(self.num_tasks)) #extra dummy task, KEEP duplicates
        z_len = (self.num_tasks + 1)*self.num_robots #each agent can finish on each task -- include dummy task, as agents can do 0 tasks
        s_len = self.num_tasks
        f_len = self.num_tasks

        flow_x = np.zeros((x_len,))
        flow_o = np.zeros((o_len,))
        flow_z = np.zeros((z_len,))
        flow_s = np.zeros((s_len,))
        flow_f = np.zeros((f_len,))

        ind_x_ak = np.reshape(np.arange(x_len), (self.num_robots, self.num_tasks + 1))
        # reshape o_akk so that o_akk[a, k+1, k'] = 1 --> agent a performs task k' immediately after task k
        ind_o_akk = np.reshape(np.arange(o_len), (self.num_robots, self.num_tasks+1, self.num_tasks))
        ind_z_ak = np.reshape(np.arange(z_len), (self.num_robots, self.num_tasks + 1))

        flow_solution = self.num_robots*np.copy(flow)

        # each agent completes its dummy task
        for a in range(self.num_robots):
            flow_x[ind_x_ak[a,0]] = 1

        ################# BEGIN DEPTH-FIRST GRAPH TRAVERSAL BY AGENT ##################
        # depth-first graph traversal for each agent
        for a in range(self.num_robots):
            cur_node = 0
            max_iter = 1000
            cur_iter = 0
            stop_condition = False
            while not stop_condition:
                out_edges = list(self.task_graph.out_edges(cur_node))
                out_edge_inds = [list(self.task_graph.edges).index(edge) for edge in out_edges]

                if len(out_edges) == 0: # if no outgoing edges, it is the agent's last task
                    flow_z[ind_z_ak[a,cur_node+1]] = 1
                    stop_condition = True

                for ind in out_edge_inds:
                    if flow_solution[ind] >= 1:
                        out_node = self.edges[ind][1]
                        flow_x[ind_x_ak[a,out_node+1]] = 1
                        flow_o[ind_o_akk[a,cur_node+1,out_node]] = 1
                        if cur_node == 0: #if robot is assigned a task, bring it to node 0 and assign it task 0
                            flow_x[ind_x_ak[a,1]] = 1
                            flow_o[ind_o_akk[a,0,0]] = 1
                        cur_node = out_node
                        flow_solution[ind] = flow_solution[ind] - 1
                        stop_condition = False
                        break
                    else:
                        stop_condition = True  # if nowhere to go, stop condition will remain true at end of loop

                if stop_condition:
                    if cur_node == 0:
                        flow_z[ind_z_ak[a,0]] = 1
                    else:
                        flow_z[ind_z_ak[a,cur_node+1]] = 1

                cur_iter += 1
                if cur_iter > max_iter:
                    #breakpoint()
                    break
        ################# END DEPTH-FIRST GRAPH TRAVERSAL BY AGENT ##################

        start_times, end_times = self.time_task_execution(flow)
        end_times_conservative = [t if t>0 else 500 for t in end_times ]
        end_times_conservative[0] = 0 #task 0 takes 0 time
        for task in range(self.num_tasks):
            flow_s[task] = start_times[task]
            flow_f[task] = end_times_conservative[task]

        # populate z vector for agents that were not assigned tasks
        for a in range(self.num_robots):
            if not flow_o[ind_o_akk[a,0,0]]:
                flow_z[ind_z_ak[a,0]] = 1


        warm_sol = self.minlp_obj.model.createSol()
        vars = self.minlp_obj.model.getVars()
        full_soln = np.concatenate((flow_x, flow_o, flow_z, flow_s, flow_f))
        for (var, var_ind) in zip(vars,range(len(full_soln))):
            self.minlp_obj.model.setSolVal(warm_sol,var,full_soln[var_ind])
        self.minlp_obj.model.setSolVal(warm_sol,self.minlp_obj.z, -1*self.reward_model.flow_cost(self.pruned_rounded_baseline_solution))
        check_result = self.minlp_obj.model.checkSol(warm_sol)
        accepted = self.minlp_obj.model.addSol(warm_sol,free=False)
        return np.zeros((x_len+o_len+z_len+s_len+f_len,))
    def solve_graph_minlp_dummy(self):
        x_len = (self.num_tasks+1)*self.num_robots # extra dummy task
        o_len = self.num_robots*((self.num_tasks+1)*(self.num_tasks)) #extra dummy task, KEEP duplicates
        z_len = (self.num_tasks + 1)*self.num_robots #each agent can finish on each task -- include dummy task, as agents can do 0 tasks
        s_len = self.num_tasks
        f_len = self.num_tasks
        self.last_minlp_solution_val = 0.0
        self.last_minlp_solution = np.zeros((x_len+o_len+z_len+s_len+f_len,))
        self.minlp_makespan = 0.0
        self.minlp_dual_bound = 1000000000
        self.minlp_primal_bound = 1000000000
        self.minlp_gap = 1000000000
        self.minlp_obj_limit = 1000000000

    def solve_graph_scipy(self):

        if self.makespan_constraint != 'cleared':
            # graph instantiation not pruned -- solve pruned graphs and choose best solution
            pruned_solutions = []
            pruned_rewards = []
            for g in self.pruned_graph_list:
                try:
                    g.solve_graph_scipy()
                except(ValueError):
                    class CustomSolution:
                        pass
                    g.last_baseline_solution = CustomSolution
                    g.last_baseline_solution.x = np.zeros((self.num_edges,))


                pruned_solutions.append(g.last_baseline_solution)
                pruned_rewards.append(-g.reward_model.flow_cost(g.last_baseline_solution.x))
            best_solution_ind = np.argmax(np.array(pruned_rewards))
            #breakpoint()

            best_flows_pruned = pruned_solutions[best_solution_ind].x
            edge_mappings = self.pruned_graph_edge_mappings_list[best_solution_ind]

            # construct best solution from a pruned graph in terms of the current graph's edges
            flows = np.zeros((len(self.edges),))
            for edge_ind in range(len(best_flows_pruned)):
                mapped_ind = edge_mappings[edge_ind]
                flows[mapped_ind] = best_flows_pruned[edge_ind]

            # save best solution in solution object
            class CustomSolution:
                pass

            self.pruned_baseline_solution = CustomSolution
            self.pruned_baseline_solution.x = flows
            self.pruned_rounded_baseline_solution = self.round_graph_solution(self.pruned_baseline_solution.x)


        if self.num_tasks < 2:
            # no valid solution possible under flow-based representation. Save a blank solution
            class CustomSolution:
                pass
            self.last_baseline_solution = CustomSolution
            self.last_baseline_solution.x = []
            self.rounded_baseline_solution = []


        self.incidence_mat = nx.linalg.graphmatrix.incidence_matrix(self.task_graph, oriented=True).A
        b = np.zeros(self.num_tasks)

        # scipy version
        # equality flow constraint
        lb2 = np.zeros(self.num_tasks-2)
        ub2 = np.ones(self.num_tasks-2)
        c2 = LinearConstraint(self.incidence_mat[1:-1,:], lb=lb2, ub=ub2)  # TODO CHANGE TO LEQ

        # inequality constraint on edge capacity
        c1 = LinearConstraint(np.eye(self.num_edges),
                               lb = np.zeros(self.num_edges),
                               ub = np.ones(self.num_edges))

        # inequality constraint on beginning flow
        c3 = LinearConstraint(self.incidence_mat[0,:],lb=[-1],ub=0)
        init_val = 0.5
        constraints_tuple = tuple(constraint for constraint in [c1,c2,c3] if constraint.A.size != 0)
        scipy_result = minimize(self.reward_model.flow_cost, np.ones(self.num_edges)*init_val, constraints=constraints_tuple)
        if scipy_result.success == False:
            while scipy_result.success==False and init_val >= 0:
                print("Re-init scipy with val ", init_val)
                scipy_result = minimize(self.reward_model.flow_cost, np.ones(self.num_edges)*init_val, constraints=constraints_tuple)
                init_val = init_val - 0.1
        self.last_baseline_solution = scipy_result
        self.rounded_baseline_solution = self.round_graph_solution(self.last_baseline_solution.x)

    def round_graph_solution(self, flows):
        ordered_nodes = list(nx.topological_sort(self.task_graph))
        #convert flows to units of whole agents
        flows_mult = flows*self.num_robots
        rounded_flows = np.zeros_like(flows)

        for curr_node in ordered_nodes[0:-1]:
            out_edges = self.task_graph.out_edges(curr_node)
            out_edge_inds = [list(self.task_graph.edges).index(edge) for edge in out_edges]
            in_edges = list(self.task_graph.in_edges(curr_node))
            in_edge_inds = [list(self.task_graph.edges).index(edge) for edge in in_edges]

            out_flows = [flows_mult[i] for i in out_edge_inds]

            # calculate total flow to be allotted
            if curr_node == 0:
                in_flow_exact = np.sum(out_flows)
                in_flow = np.around(in_flow_exact)
            else:
                in_flows = [rounded_flows[i] for i in in_edge_inds]
                in_flow = np.sum(in_flows)

            # round out flows to nearest digit
            candidate_out_flows = np.around(out_flows)

            # if we rounded up too much and created more outflow than inflow:
            while (np.sum(candidate_out_flows) > in_flow):
                diff = in_flow - np.sum(candidate_out_flows)
                rounding_diffs = candidate_out_flows - out_flows
                for i in range(len(rounding_diffs)):
                    if candidate_out_flows[i] == 0.0:
                        rounding_diffs[i] = -10000000
                if diff < 0:
                    decrease_ind = np.argmax(rounding_diffs)
                    candidate_out_flows[decrease_ind] = candidate_out_flows[decrease_ind] - 1

            #populate new flows into vector
            for (f, i) in zip(candidate_out_flows, out_edge_inds):
                if f < 0.0:
                    pass
                rounded_flows[i] = f

        return rounded_flows/self.num_robots

    def solve_graph_greedy(self):

        if self.makespan_constraint != 'cleared':
            # graph instantiation not pruned -- solve pruned graphs and choose best solution
            pruned_solutions = []
            pruned_rewards = []


            for g in self.pruned_graph_list:
                g.solve_graph_greedy()
                pruned_solutions.append(g.last_greedy_solution)
                pruned_rewards.append(-g.reward_model.flow_cost(g.last_greedy_solution))


            best_solution_ind = np.argmax(np.array(pruned_rewards))
            best_flows_pruned = pruned_solutions[best_solution_ind]
            edge_mappings = self.pruned_graph_edge_mappings_list[best_solution_ind]

            # construct best solution from a pruned graph in terms of the current graph's edges
            flows = np.zeros((len(self.edges),))
            for edge_ind in range(len(best_flows_pruned)):
                mapped_ind = edge_mappings[edge_ind]
                flows[mapped_ind] = best_flows_pruned[edge_ind]

            # save best solution
            self.pruned_greedy_solution = flows
            self.pruned_rounded_greedy_solution = self.round_graph_solution(self.pruned_greedy_solution)

        initial_flow = 1.0
        self.last_greedy_solution = np.zeros((self.num_edges,))

        ordered_nodes = list(nx.topological_sort(self.task_graph))

        for curr_node in ordered_nodes[:-1]:
            out_edges = self.task_graph.out_edges(curr_node)
            if len(out_edges) > 0:
                out_edge_inds = [list(self.task_graph.edges).index(edge) for edge in out_edges]
                in_edges = list(self.task_graph.in_edges(curr_node))
                in_edge_inds = [list(self.task_graph.edges).index(edge) for edge in in_edges]
                num_edges = len(out_edges)
                if num_edges == 0:
                    pass
                    #breakpoint()
                # make cost function handle that takes in edge values and returns rewards
                def node_reward(f, out_edge_inds, out_neighbors):
                    sort_mapping = np.argsort(out_edge_inds)
                    last_ind = -1
                    arrays_list = []
                    for mapping_ind in sort_mapping:
                        out_edge_ind = out_edge_inds[mapping_ind]
                        if last_ind + 1 != out_edge_ind:
                            arrays_list.append(np.array(self.last_greedy_solution[last_ind+1:out_edge_ind]))
                        arrays_list.append(np.atleast_1d(f[mapping_ind]))
                        last_ind = out_edge_ind
                    if last_ind != self.num_edges-1:
                        arrays_list.append(np.array(self.last_greedy_solution[last_ind+1:]))

                    input_flows = np.concatenate(arrays_list)
                    rewards = -1*self.reward_model._nodewise_optim_cost_function(input_flows)
                    relevant_costs = np.array(rewards[out_neighbors])
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
                    cand_flow_rewards.append(node_reward(cand_flow,out_edge_inds,[int(edge[1]) for edge in out_edges]))

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
                    gradient_t = gradient_func(last_state, out_edge_inds, [int(edge[1]) for edge in out_edges])
                    if np.isnan(gradient_t).any():
                        gradient_t = np.zeros_like(gradient_t)
                        print("WARNING: GRADIENT WAS NAN, REPLACED WITH ZERO")
                    # TODO: project gradient onto hyperplane that respects constraints
                    # FOR NOW: just normalize new state such that it is valid

                    # take a step along that vector direction
                    new_cand_state = last_state + dt*gradient_t
                    for k in range(num_edges):
                        if new_cand_state[k] <= 0:
                            new_cand_state[k] = 0.00001
                    last_state = incoming_flow*new_cand_state/np.sum(new_cand_state)



                # update self.last_greedy_solution
                for (edge_i, new_flow) in zip(out_edge_inds,last_state):
                    self.last_greedy_solution[edge_i] = new_flow
                    if np.isnan(new_flow):
                        print("GREEDY SOLUTION FLOW IS NAN")
                        #breakpoint()

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
        incomplete_nodes = []
        while len(frontier_nodes) > 0:
            current_node = frontier_nodes.pop(0)
            incoming_edges = [list(e) for e in self.task_graph.in_edges(current_node)]
            incoming_edge_inds = [self.reward_model.edges.index(e) for e in incoming_edges]
            if len(incoming_edges) > 0:
                if np.array([flow[incoming_edge_inds[i]]<=0.000001 for i in range(len(incoming_edges))]).all():
                    task_start_times[int(current_node)] = 0.0
                    incomplete_nodes.append(current_node)
                else:
                    task_start_times[int(current_node)] = max([task_finish_times[int(incoming_edges[i][0])] for i in range(len(incoming_edges)) if not incoming_edges[i][0] in np.array(incomplete_nodes)])
            else:
                task_start_times[int(current_node)] = 0
            task_finish_times[int(current_node)] = task_start_times[int(current_node)] + self.task_times[int(current_node)]
            for n in self.task_graph.neighbors(current_node):
                if n not in frontier_nodes:
                    frontier_nodes.append(n)

        for node in incomplete_nodes:
            task_finish_times[int(node)] = 0.0
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
        info_dict = self.translate_minlp_objective(minlp_objective)
        breakpoint()
        import pdb; pdb.set_trace()

    def translate_minlp_objective(self, x):
        info_dict = {}
        x_ak, o_akk, z_ak, s_k, f_k = self.minlp_obj.partition_x(x)
        # x_ak organized by agent
        x_ak = np.reshape(np.array(x_ak), (self.num_robots, self.num_tasks + 1)) # reshape so each row contains x_ak for agent a
        x_dummy = x_ak[:,0]
        x_ak = x_ak[:,1:]
        task_coalitions = []#np.zeros((self.num_tasks,)) # list of coalition size assigned to each task
        for t in range(self.num_tasks):
            task_coalitions.append(np.sum(x_ak[:,t]))
        print("task coalitions: ",task_coalitions)
        info_dict['task coalitions'] = str(task_coalitions)
        o_akk = np.atleast_3d(np.reshape(o_akk,(self.num_robots,self.num_tasks+1, self.num_tasks)))
        order_string = []
        for a in range(self.num_robots):
            for j in range(self.num_tasks):
                for k in range(self.num_tasks):
                    if(o_akk[a,j+1,k]>0.99):
                        print("agent %d performs task %d before task %d" %(a,j,k))
                        order_string.append("agent %d performs task %d before task %d" %(a,j,k))
        info_dict['task order'] = order_string
        tasks_ordered = np.argsort(np.array(f_k))
        time_string = []
        for t in tasks_ordered:
            print("Time ", f_k[t],": task %d completed by %d agents" % (t, task_coalitions[t]))
            time_string.append("Time " + str(f_k[t]) + ": task %d completed by %d agents" % (t, task_coalitions[t]))
        info_dict['task times'] = time_string

        completed_task_times = [f_k[k] for k in range(self.num_tasks) if task_coalitions[k]>0.0]
        if len(completed_task_times) > 0:
            info_dict['makespan'] = np.max(completed_task_times)
        else:
            info_dict['makespan'] = 0.0
        return info_dict
