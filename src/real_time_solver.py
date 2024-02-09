from taskgraph import TaskGraph
import networkx as nx
import numpy as np
import copy
import toml
import matplotlib.pyplot as plt


class RealTimeSolver:

    def __init__(self, taskgraph_arg_dict):
        # initialize task graph with arguments
        # keep track of free and busy agents, completed and incomplete tasks, reward
        # model, etc
        self.original_args = taskgraph_arg_dict
        self.original_task_graph = TaskGraph(**taskgraph_arg_dict)
        self.original_num_tasks = self.original_task_graph.num_tasks
        self.original_num_robots = self.original_task_graph.num_robots
        self.task_completed = [False for _ in range(self.original_num_tasks)]
        self.task_rewards = [0 for _ in range(self.original_num_tasks)]
        self.agent_functioning = [True for _ in range(self.original_num_robots)]
        self.agent_free = [True for _ in range(self.original_num_robots)]
        self.current_time = 0.0
        self.current_step = 0

        # solve task graph
        # TODO: will we want to just do one solution at a time? probably
        self.original_task_graph.solve_graph_scipy()
        # save pruned rounded NLP solution as current solution -- set of flows over edges
        self.current_solution = self.original_task_graph.pruned_rounded_baseline_solution
        self.current_nodewise_rewards = -self.original_task_graph.reward_model._nodewise_optim_cost_function(
            self.current_solution, debug=False)
        self.original_solution = copy.deepcopy(self.current_solution)
        self.original_rewards = copy.deepcopy(self.current_nodewise_rewards)

        self.original_agent_assignment, self.original_start_times, self.original_finish_times = self.get_assignment(self.original_task_graph,
                                                                                                                    self.current_solution)
        self.original_ordered_finish_times = np.sort(self.original_finish_times)
        fin_time_inds = np.argsort(self.original_finish_times)
        self.original_ordered_finish_times_inds = fin_time_inds[self.original_ordered_finish_times > 0]
        self.original_ordered_finish_times = self.original_ordered_finish_times[self.original_ordered_finish_times > 0]

        self.current_agent_assignment = copy.deepcopy(self.original_agent_assignment)
        self.current_start_times = copy.deepcopy(self.original_start_times)
        self.current_finish_times = copy.deepcopy(self.original_finish_times)
        self.current_ordered_finish_times = copy.deepcopy(self.original_ordered_finish_times)
        self.current_ordered_finish_times_inds = copy.deepcopy(self.original_ordered_finish_times_inds)

        self.current_actual_and_expected_rewards = copy.deepcopy(self.current_nodewise_rewards)

        print(f"original agent assignments: {self.current_agent_assignment}")
        print(f"original finish times: {self.current_finish_times}")
        print(f"original task order: {self.current_ordered_finish_times_inds}")
        self.ghost_node_param_dict = {}

        self.nodepos_dict = self.draw_original_taskgraph(self.current_solution)

        self.all_taskgraphs = [self.original_task_graph]

    def step(self, completed_tasks, inprogress_tasks, inprogress_task_times, inprogress_task_coalitions, rewards,
             free_agents, time):
        # inputs:
        # completed_tasks -- list of task IDs of completed tasks
        # inprogress_tasks -- list of task IDs of tasks currently in progress
        # inprogress_task_times -- list of time already spent (s) working on each inprogress task. order same as inprogress_tasks
        # inprogress_task_coalitions -- list of lists, where list i is the set of agents currently working on task inprogress task i (same order as inprogress_tasks)
        # TODO do we need input of agents working on each of these in progress tasks? or can we get that from initial assignment??
        # rewards -- list of rewards from completed tasks, same order as completed tasks
        # free_agents -- list of agent IDs of free agents that have just completed the tasks
        # time -- float current time

        # takes in an update from graph manager node on current graph status.
        # uses update to generate a new task allocation plan
        # NOTE: best to keep everything in terms of the original task graph
        # -- translate to new graph, solve, and then immediately translate back
        print(f"-------------------NEW STEP {self.current_step} ----------------------")
        print(f"Completed tasks: {completed_tasks}")
        print(f"Completed task rewards: {rewards}")
        print(f"Free agents: {free_agents}")
        print(f"Current time: {time}")

        task_it = 0
        for task in completed_tasks:
            self.task_completed[task] = True
            self.task_rewards[task] = rewards[task_it]
            task_it += 1

        # initial source node is automatically always completed when step is called
        self.task_completed[0] = True
        self.task_rewards[0] = 0

        for task in completed_tasks:
            if task != 0:
                in_edges = [e for e in self.original_task_graph.edges if e[1] == task]
                for edge in in_edges:
                    if not self.task_completed[edge[0]]:
                        self.task_completed[edge[0]] = True
                        self.task_rewards[edge[0]] = 0.0
                        print(f"SUBSEQUENT TASK {edge[1]} COMPLETED, MARKING TASK {edge[0]} AS COMPLETE")

        self.current_time = time

        new_task_graph, new_to_old_node_mapping = self.create_new_taskgraph(inprogress_tasks, inprogress_task_times,
                                                                            inprogress_task_coalitions)

        # solve flow problem
        new_task_graph.solve_graph_scipy()

        # new flow solution
        new_flow_solution = new_task_graph.pruned_rounded_baseline_solution

        if len(new_task_graph.edges)==1:
            best_flow = new_flow_solution
            best_reward = -100000000
            for i in range(self.original_num_robots):
                proposed_flow = [i/self.original_num_robots]
                proposed_r = -new_task_graph.reward_model.flow_cost(proposed_flow)
                if proposed_r > best_reward:
                    best_flow = proposed_flow
                    best_reward = proposed_r
            new_flow_solution = best_flow
        new_flow_solution_old_edges = [0 for _ in range(len(self.original_task_graph.edges))]
        updated_inds = []
        for ind in range(len(new_flow_solution)):
            edge = new_task_graph.edges[ind]
            old_edge = (new_to_old_node_mapping[edge[0]], new_to_old_node_mapping[edge[1]])
            if -1 not in old_edge:  # edge exists in original graph
                old_edge_ind = self.original_task_graph.edges.index(old_edge)  # raises ValueError if not in list
                updated_inds.append(old_edge_ind)
                new_flow_solution_old_edges[old_edge_ind] = new_flow_solution[ind]

            # if edge is newly connected to source, ensure destination node has correct incoming flow
            if edge[0] < new_task_graph.source_node_info_dict['num_source_nodes']:
                new_graph_in_edges = [e for e in new_task_graph.edges if e[1]==edge[1]]
                new_graph_in_edge_inds = [new_task_graph.edges.index(e) for e in new_graph_in_edges]
                new_graph_node_inflow = 0
                for in_edge_ind in new_graph_in_edge_inds:
                    new_graph_node_inflow += new_flow_solution[in_edge_ind]
                old_graph_in_edges = [e for e in self.original_task_graph.edges if e[1] == old_edge[1]]
                old_graph_in_edge_inds = [self.original_task_graph.edges.index(e) for e in old_graph_in_edges]
                old_graph_node_inflow = 0
                for in_edge_ind in old_graph_in_edge_inds:
                    old_graph_node_inflow += new_flow_solution_old_edges[in_edge_ind]
                # NOTE: the below difference correction can result in flows on the old graph that violate the
                # flow constraints. However, no flow constraints will be violated in real time -- limited to past
                # corrections necessary for calculating results
                if new_graph_node_inflow != old_graph_node_inflow:
                    diff = new_graph_node_inflow - old_graph_node_inflow
                    new_flow_solution_old_edges[old_graph_in_edge_inds[0]] += diff
                    updated_inds.append(old_graph_in_edge_inds[0])

        for ind in range(len(self.current_solution)):
            if not (ind in updated_inds):
                new_flow_solution_old_edges[ind] = self.current_solution[ind]

        # NOTE: OLD FLOW REWARD is not a valid metric. Its nodewise rewards need not equal the actual nodewise rewards.
        # this is due to inability to account for changes in rewards plugged into dependency functions over real-time iterations
        old_flow_reward = self.original_task_graph.reward_model.flow_cost(self.current_solution)
        new_flow_reward = self.original_task_graph.reward_model.flow_cost(new_flow_solution_old_edges)
        new_flow_reward_new_graph = new_task_graph.reward_model.flow_cost(new_flow_solution)
        print("old graph reward calc:")
        old_flow_reward_nodewise = -self.original_task_graph.reward_model._nodewise_optim_cost_function(
            self.current_solution, debug=False)
        new_flow_reward_nodewise = -self.original_task_graph.reward_model._nodewise_optim_cost_function(
            new_flow_solution_old_edges, debug=False)  # translate new flow solution to original task graph and save
        print("new graph reward calc:")
        new_flow_reward_nodewise_new_graph = -new_task_graph.reward_model._nodewise_optim_cost_function(
            new_flow_solution, debug=False)
        # NOTE: after this, the new flow solution may not be valid under flow constraints
        #print(f"Old flow solution: {self.current_solution}")
        #print(f"New flow solution: {new_flow_solution}")
        #print(f"New flow solution old edges: {new_flow_solution_old_edges}")
        print("Solution changes this iteration:")
        for i in range(len(new_flow_solution_old_edges)):
            if self.current_solution[i] - new_flow_solution_old_edges[i] != 0:
                print(f"{self.original_task_graph.edges[i]}: {self.current_solution[i]} --> {new_flow_solution_old_edges[i]}")
        # if self.current_step == 9:
        #     import pdb; pdb.set_trace()
        print(f"Original total reward: {-self.original_task_graph.reward_model.flow_cost(self.original_solution)}")
        #print(f"Old total reward: {-old_flow_reward}")
        #print(f"New total reward: {-new_flow_reward}")
        print(f"Old actual and expected total: {np.sum(self.current_actual_and_expected_rewards)}")
        current_actual_and_predicted = []
        for task in range(self.original_num_tasks):
            if self.task_completed[task]:
                current_actual_and_predicted.append(self.task_rewards[task])
            else:
                new_task_name = new_to_old_node_mapping.index(task)
                current_actual_and_predicted.append(new_flow_reward_nodewise_new_graph[new_task_name])
        print(f"Current actual reward + predicted: {np.sum(current_actual_and_predicted)}")
        # print(new_flow_reward_new_graph)
        print(old_flow_reward_nodewise)
        print(new_flow_reward_nodewise)
        print(f"current actual and predicted rewards: {current_actual_and_predicted}")
        #print(new_flow_reward_nodewise_new_graph)

        # old flow solution into new graph
        old_flow_new_graph = [0.0 for _ in range(len(new_flow_solution))]
        for (edge, edge_id) in zip(new_task_graph.edges, range(len(new_task_graph.edges))):
            old_edge = (new_to_old_node_mapping[edge[0]], new_to_old_node_mapping[edge[1]])
            if old_edge in self.original_task_graph.edges:
                old_edge_ind = self.original_task_graph.edges.index(old_edge)
                old_flow_new_graph[edge_id] = self.current_solution[old_edge_ind]


        self.draw_new_taskgraph(new_task_graph, new_to_old_node_mapping, new_flow_solution)
        self.current_solution = new_flow_solution_old_edges

        new_agent_assignment, new_start_times, new_finish_times = self.get_assignment(new_task_graph, new_flow_solution)
        for i in range(len(new_agent_assignment)):
            old_node_name = new_to_old_node_mapping[i]
            self.current_agent_assignment[old_node_name] = new_agent_assignment[i]
            self.current_start_times[old_node_name] = new_start_times[i] + self.current_time
            self.current_finish_times[old_node_name] = new_finish_times[i] + self.current_time

        self.current_ordered_finish_times = np.sort(self.current_finish_times)
        fin_time_inds = np.argsort(self.current_finish_times)
        self.current_ordered_finish_times_inds = fin_time_inds[self.current_ordered_finish_times > self.current_time]
        self.current_ordered_finish_times = self.current_ordered_finish_times[self.current_ordered_finish_times > self.current_time]
        print(f"new agent assignments: {self.current_agent_assignment}")
        print(f"new finish times: {self.current_finish_times}")
        print(f"new task order: {self.current_ordered_finish_times_inds}")
        self.current_actual_and_expected_rewards = current_actual_and_predicted
        self.all_taskgraphs.append(new_task_graph)
        if np.all(new_flow_solution == 0.):
            for task in [t for t in range(self.original_num_tasks) if not self.task_completed[t]]:
                new_task_ind = new_to_old_node_mapping.index(task)
                self.task_rewards[task] = new_flow_reward_nodewise_new_graph[new_task_ind]
            return True  # FINISHED
        return False  # NOT YET FINISHED


    def sim_step(self):
        # simulates a step forward in time, calls step function with updated graph
        # will be used to implement disturbances into the simulator to test disturbance
        # response/ robustness in stripped down simulator
        for task in self.current_ordered_finish_times_inds:
            if not self.task_completed[task]:
                task_completed = task
                time = self.current_finish_times[task]
                break

        if len(self.current_ordered_finish_times_inds)==0 or len(self.all_taskgraphs[self.current_step].edges)==1:
            if len(self.all_taskgraphs[self.current_step].edges)==1 and len(self.current_ordered_finish_times_inds) != 0 :
                self.task_rewards[task_completed] = self.current_actual_and_expected_rewards[task_completed]
            print("REAL TIME REALLOCATION COMPLETE: NO FURTHER TASKS")
            return True

        inprogress_tasks = []
        inprogress_task_times = []
        inprogress_coalitions = []
        for task in range(self.original_num_tasks):
            if time > self.current_start_times[task] and time < self.current_finish_times[task] and task != task_completed:
                inprogress_tasks.append(task)
                inprogress_task_times.append(time - self.current_start_times[task])
                inprogress_coalitions.append(self.current_agent_assignment[task])
        # for now, get exact expected reward
        all_rewards = -self.original_task_graph.reward_model._nodewise_optim_cost_function(
            self.original_task_graph.pruned_rounded_baseline_solution)
        reward = self.current_actual_and_expected_rewards[task_completed]
        #reward = all_rewards[task_completed]
        #print(f"SAVING REWARD: {reward}")
        #print(f"WOULDA BEEN: {self.current_actual_and_expected_rewards[task_completed]}")
        #print(f"WOULDA BEEN: {all_rewards[task_completed]}")
        free_agents = self.current_agent_assignment[task_completed]
        self.current_step += 1

        self.step([task_completed], inprogress_tasks, inprogress_task_times, inprogress_coalitions, [reward],
                  free_agents, time)
        if self.current_step >= self.original_num_tasks:
            import pdb; pdb.set_trace()
        return self.current_step >= self.original_num_tasks # return true if DONE

        # need list of tasks completed, list of rewards of those tasks, list of free agents
        # that have just finished completed tasks

    def create_new_taskgraph(self, inprogress_tasks, inprogress_task_times, inprogress_task_coalitions):
        # creates a new task graph from old task graph parameters
        # revises reward model, replacing completed tasks with constant reward values in the influence
        # functions of future tasks, and replacing in-progress tasks with expected reward values in the
        # same manner
        # removes completed and in-progress tasks from the graph
        # maintains a map from new nodes/edges to old
        # SIDE NOTE how do we specify multiple source nodes? have to change scipy constraints

        # create copy of original task graph args
        new_args = copy.deepcopy(self.original_args)
        # fix any ordering discrepancies
        new_args['edges'] = copy.deepcopy(
            self.original_task_graph.reward_model.edges)  # list of lists rather than list of tuples
        new_args['dependency_params'] = copy.deepcopy(self.original_task_graph.dependency_params)
        new_args['dependency_types'] = copy.deepcopy(self.original_task_graph.dependency_types)

        edges_to_delete_inds = []  # each list entry i is the id of an edge that we need to delete
        for task in range(self.original_num_tasks):
            if self.task_completed[task]:
                # create list of completed task outgoing edge inds
                outgoing_edges = [list(e) for e in self.original_task_graph.task_graph.out_edges(task)]
                outgoing_edge_inds = [self.original_task_graph.reward_model.edges.index(e) for e in outgoing_edges]
                incoming_edges = [list(e) for e in self.original_task_graph.task_graph.in_edges(task)]
                incoming_edge_inds = [self.original_task_graph.reward_model.edges.index(e) for e in incoming_edges]
                concat_edge_inds = outgoing_edge_inds + incoming_edge_inds
                for edge_id in concat_edge_inds:
                    if edge_id not in edges_to_delete_inds:
                        edges_to_delete_inds.append(edge_id)

        edges_to_delete_inds.sort(reverse=True)  # sort edge ids from highest to lowest to delete them
        for edge_id in edges_to_delete_inds:  # delete edges from taskgraph args
            new_args['edges'].pop(edge_id)
            new_args['dependency_params'].pop(edge_id)
            new_args['dependency_types'].pop(edge_id)

        # which nodes have incoming flow from a completed task that we need to consider?
        nodes_with_deleted_incoming_flow = []
        #amount_deleted_incoming_flow = []
        for i in [j for j in range(len(self.task_completed)) if self.task_completed[j]]:
            out_neighbors_i = [edge[1] for edge in self.original_task_graph.edges if edge[0] == i]
            for nbr in out_neighbors_i:
                if (nbr not in inprogress_tasks) and not self.task_completed[nbr]:
                    if self.current_solution[self.original_task_graph.edges.index((i, nbr))] > 0.00001:
                        nodes_with_deleted_incoming_flow.append(nbr)
                    #amount_deleted_incoming_flow.append(self.current_solution[self.original_task_graph.edges.index((i, nbr))])

        # how many new nodes total
        num_new_sources = 1 + len(inprogress_tasks) #+ len(nodes_with_deleted_incoming_flow)

        # add new source node that connects to nodes without incoming edges that are NOT in progress
        nodes_to_keep = [task for task in range(self.original_num_tasks) if not self.task_completed[task]]
        nodes_to_connect_to_source = []
        nodes_formerly_connected_to_source = [edge[1] for edge in self.original_task_graph.edges if
                                              (edge[0] == 0 and edge[1] in nodes_to_keep and edge[1] not in inprogress_tasks)]
        for task in nodes_to_keep:
            node_has_in_neighbors = False
            if task not in inprogress_tasks:
                for edge_id in new_args['edges']:  # iterate through edges we're keeping (w/ original names)
                    if task == edge_id[1]:
                        node_has_in_neighbors = True
            if (not node_has_in_neighbors) or task in nodes_formerly_connected_to_source:
                nodes_to_connect_to_source.append(task)

        # add ONE source node that connects to all nodes without incoming neighbors
        # all free agents on this source node
        # TODO: NEED to generalize this to several new source nodes w/ different agent capacities on each
        # TODO: for both nodes and edges leaving those nodes. all neighbors to new source nodes must be
        # TODO: modified to ensure that their rewards won't be modified by this artificial source node
        # TODO: source nodes corresponding to robot teams working on a task (in progress) must have appopriate params so as not to impact new neighbors
        # they will connect only to the task they are working on
        # TODO: the duration of in-progress tasks should be modified to reflect progress (time)
        # TODO: precisely which nodes does the source node with free robots connect?

        # generate mapping of new taskgraph node ids to old taskgraph node ids
        # entry of -1 corresponds to NEW node -- no corresponding node on old graph
        new_to_old_node_mapping = [-1] * num_new_sources + [task for task in range(self.original_num_tasks) if
                                                            not self.task_completed[task]]

        # TODO this only works if each outgoing neighbor has only one incoming node that gets completed this step --
        #  this is probably a valid assumption most of the time? Will only be violated if we want to do batch processing of steps
        ghost_node_param_dict = {}

        for task in [t for t in range(1, self.original_num_tasks) if self.task_completed[t]]:
            outgoing_edges = [e for e in self.original_task_graph.edges if e[0] == task]
            outgoing_edge_inds = [self.original_task_graph.edges.index((e[0], e[1])) for e in outgoing_edges]
            outgoing_neighbors = [e[1] for e in outgoing_edges]
            for (nbr, nbr_edge) in zip(outgoing_neighbors, outgoing_edge_inds):
                if nbr in new_to_old_node_mapping:  # if the outgoing nbr exists in the new graph (i.e. it is not a completed task)
                    # list of tuples of (influence_type, influence_params, reward)
                    new_nbr_name = new_to_old_node_mapping.index(nbr)
                    new_entry = (self.original_task_graph.dependency_types[nbr_edge],
                                 self.original_task_graph.dependency_params[nbr_edge],
                                 self.task_rewards[task])
                    if str(new_nbr_name) in ghost_node_param_dict.keys():
                        old_entry = copy.deepcopy(ghost_node_param_dict[str(new_nbr_name)])
                        if new_entry not in old_entry:
                            old_entry.append(new_entry)
                            ghost_node_param_dict[str(new_nbr_name)] = old_entry
                    else:
                        ghost_node_param_dict[str(new_nbr_name)] = [new_entry]

        new_args['ghost_node_param_dict'] = ghost_node_param_dict

        # add source nodes for each in-progress task
        # add number of agents at each source node that are working on the corresponding task

        # rename old edges
        for edge_id in range(len(new_args['edges'])):
            new_args['edges'][edge_id] = [new_to_old_node_mapping.index(new_args['edges'][edge_id][0]),
                                          new_to_old_node_mapping.index(new_args['edges'][edge_id][1])]

        # add new edges
        cur_source = 1  # start on 1, leave 0 for new source node for free agents
        for node in (inprogress_tasks):  # + nodes_with_deleted_incoming_flow):
            new_node_name = new_to_old_node_mapping.index(node)
            new_args['edges'] = [[cur_source, new_node_name]] + new_args['edges']
            new_args['dependency_types'] = ['polynomial'] + new_args['dependency_types']
            new_args['dependency_params'] = [[0., 0., 0.]] + new_args['dependency_params']
            cur_source += 1

        # if the only nodes to connect to source are inprogress nodes, then connect the source to
        # all outgoing neighbors of inprogress nodes
        if np.all([node in inprogress_tasks for node in (nodes_to_connect_to_source +
                  [node for node in nodes_with_deleted_incoming_flow if node not in nodes_to_connect_to_source])]):
            for task in inprogress_tasks:
                new_task_name = new_to_old_node_mapping.index(task)
                out_nbrs = [e[1] for e in new_args['edges'] if e[0] == new_task_name]
                for out_nbr in out_nbrs:
                    if out_nbr not in nodes_to_connect_to_source:
                        nodes_to_connect_to_source.append(new_to_old_node_mapping[out_nbr])

        nodes_to_connect_to_source_new_names = []
        for node in (nodes_to_connect_to_source + [node for node in nodes_with_deleted_incoming_flow if node not in nodes_to_connect_to_source]):
            if node not in inprogress_tasks:
                new_node_name = new_to_old_node_mapping.index(node)
                nodes_to_connect_to_source_new_names.append(new_node_name)
                new_args['edges'] = [[0, new_node_name]] + new_args['edges']
                new_args['dependency_types'] = ['polynomial'] + new_args['dependency_types']
                new_args['dependency_params'] = [[0., 0., 0.]] + new_args['dependency_params']

        new_edges_old_names = [[new_to_old_node_mapping[e[0]], new_to_old_node_mapping[e[1]]] for e in
                               new_args['edges']]

        source_neighbors = nodes_to_connect_to_source_new_names + [new_to_old_node_mapping.index(t) for t in
                                                                   inprogress_tasks]

        # remove tasks from taskgraph
        new_args['coalition_params'] = [[0., 0., 0.]] * num_new_sources + [new_args['coalition_params'][i] for i in
                                                                           range(self.original_num_tasks) if
                                                                           not self.task_completed[i]]
        new_args['coalition_types'] = ['null'] * num_new_sources + [new_args['coalition_types'][i] for i in
                                                                    range(self.original_num_tasks) if
                                                                    not self.task_completed[i]]
        new_args['aggs'] = ['or'] * num_new_sources + [new_args['aggs'][i] for i in range(self.original_num_tasks) if
                                                       not self.task_completed[i]]
        if new_args['nodewise_coalition_influence_agg_list'] is not None:
            new_args['nodewise_coalition_influence_agg_list'] = ['sum'] * num_new_sources + [
                new_args['nodewise_coalition_influence_agg_list'][i] for i in range(self.original_num_tasks) if
                not self.task_completed[i]]
        for task in source_neighbors:  # ensure tasks with incoming edges from source nodes that are effected by ghost nodes function as identity
            if new_args['nodewise_coalition_influence_agg_list'] is not None:
                if new_args['nodewise_coalition_influence_agg_list'][task] == 'product' and task in [int(k) for k in
                                                                                                     ghost_node_param_dict.keys()]:
                    if new_args['aggs'][task] == 'and':
                        # find new edge ind
                        in_edge = [list(e) for e in new_args['edges'] if e[1] == task]
                        if len(in_edge) > 1:
                            import pdb;
                            pdb.set_trace()
                        in_edge = in_edge[0]
                        in_edge_ind = new_args['edges'].index(in_edge)
                        new_args['dependency_params'][in_edge_ind] = [1.0, 0.0, 0.0]
                else:
                    new_args['nodewise_coalition_influence_agg_list'][task] = 'sum'

        new_args['task_times'] = [0.] * num_new_sources + [self.original_task_graph.task_times[i] for i in range(self.original_num_tasks) if not self.task_completed[i]]
        # adjust in-progress task durations to reflect current time
        for task in inprogress_tasks:
            new_node_name = new_to_old_node_mapping.index(task)
            new_args['task_times'][new_node_name] = new_args['task_times'][new_node_name] - inprogress_task_times[inprogress_tasks.index(task)]

        new_args['num_tasks'] = len(new_to_old_node_mapping)

        node_capacities = [0.]
        for coalition in inprogress_task_coalitions:
            node_capacities.append(len(coalition) / self.original_num_robots)
        total_inprogress_coalitions = np.sum(np.array(node_capacities))
        node_capacities[0] = 1 - total_inprogress_coalitions

        node_capacities = node_capacities #  + amount_deleted_incoming_flow

        source_node_info_dict = {'num_source_nodes': num_new_sources,
                                 'node_capacities': node_capacities,
                                 'in_progress': [new_to_old_node_mapping.index(t) for t in inprogress_tasks]}

        new_args['source_node_info_dict'] = source_node_info_dict
        new_args['makespan_constraint'] = self.original_task_graph.makespan_constraint - self.current_time

        args_file = 'real_time_debug/new_args.toml'
        with open(args_file, "w") as f:
            toml.dump(new_args, f)

        # create new taskgraph with args
        new_task_graph = TaskGraph(**new_args)

        return new_task_graph, new_to_old_node_mapping

    def get_assignment(self, taskgraph, flow):
        ordered_nodes = list(nx.topological_sort(taskgraph.task_graph))
        frontier_nodes = []
        task_start_times = np.zeros((taskgraph.num_tasks,))
        task_finish_times = np.zeros((taskgraph.num_tasks,))
        agent_assignments = [[] for _ in range(taskgraph.num_tasks)]
        temp_agent_assignments = [[] for _ in range(taskgraph.num_tasks)]
        nodelist = list(range(taskgraph.num_tasks))
        frontier_nodes.append(nodelist[0])
        incomplete_nodes = []
        for current_node in ordered_nodes:
            #print(f"PROCESSING NODE {current_node}")
            incoming_edges = [list(e) for e in taskgraph.task_graph.in_edges(current_node)]
            incoming_edge_inds = [taskgraph.reward_model.edges.index(e) for e in incoming_edges]
            #print(f"incoming edges: {incoming_edges}")
            if len(incoming_edges) > 0:
                # if no incoming flow, no assignment, task time of 0
                if np.array([flow[incoming_edge_inds[i]] <= 0.000001 for i in range(len(incoming_edges))]).all():
                    task_start_times[int(current_node)] = 0.0
                    incomplete_nodes.append(current_node)
                    #print("no incoming flow: task is not completed")
                else:
                    task_start_times[int(current_node)] = max(
                        [task_finish_times[int(incoming_edges[i][0])] for i in range(len(incoming_edges)) if
                         not incoming_edges[i][0] in np.array(incomplete_nodes)])
                    for ind in incoming_edge_inds:
                        num_agents = int(round(flow[ind] * taskgraph.num_robots))
                        edge = taskgraph.reward_model.edges[ind]
                        #print(f"{num_agents} flowing from edge {edge}")
                        incoming_node = edge[0]
                        # TODO make the following line POP rather than just grab
                        agent_list = []
                        for _ in range(num_agents):
                            agent_list.append(temp_agent_assignments[incoming_node].pop(0))
                        temp_agent_assignments[current_node] += agent_list
                        agent_assignments[current_node] += agent_list

            else:
                #print("at start node")
                task_start_times[int(current_node)] = 0
                agent_assignments[int(current_node)] = [i for i in range(taskgraph.num_robots)]
                temp_agent_assignments[int(current_node)] = [i for i in range(taskgraph.num_robots)]

            task_finish_times[int(current_node)] = task_start_times[int(current_node)] + taskgraph.task_times[
                int(current_node)]

        for node in incomplete_nodes:
            task_finish_times[int(node)] = 0.0
        return agent_assignments, task_start_times, task_finish_times

    def draw_original_taskgraph(self, flow_solution):

        sorted_nodes = list(nx.topological_sort(self.original_task_graph.task_graph))
        num_frontiers = int(np.floor(np.log(self.original_num_tasks)))
        frontier_width = int(np.floor(self.original_num_tasks / num_frontiers))
        frontiers_list = []
        for i in range(num_frontiers):
            frontiers_list.append(sorted_nodes[i * frontier_width:(i + 1) * frontier_width])
        leftovers = sorted_nodes[num_frontiers * frontier_width:]
        if len(leftovers) > 0:
            frontiers_list.append(leftovers)

        # generate node positions
        node_pos = {}
        x_tot = 8
        y_tot = 6
        n_frontiers = len(frontiers_list) + 1
        node_pos[0] = (0,y_tot/2)
        frontier_ct = 1
        for frontier in frontiers_list:
            x_pos = frontier_ct*(1/n_frontiers)*x_tot
            frontier_ct += 1
            n_nodes = len(frontier)
            y_interval = y_tot/n_nodes
            node_ct = 1
            for node in frontier:
                node_pos[node] = (x_pos + (np.random.random()-0.5)*2,y_tot-node_ct*y_interval)
                node_ct += 1
        #node_pos = nx.kamada_kawai_layout(self.original_task_graph.task_graph, scale=5)
        node_pos = nx.shell_layout(self.original_task_graph.task_graph, scale=5)
        nodewise_rewards = self.original_task_graph.reward_model._nodewise_optim_cost_function(flow_solution)
        label_dict = {}
        for i in range(self.original_num_tasks):
            label_dict[i] = str(i) + '\n' + "{:.2f}".format(-nodewise_rewards[i])
        # nx.draw_networkx_labels(nx_task_graph, labels=label_dict)
        graph_img_file = 'real_time_debug/old_graph.jpg'
        fig, ax = plt.subplots()
        nx.draw(self.original_task_graph.task_graph, labels=label_dict, pos=node_pos, ax=ax, node_size=1200, alpha=0.4)
        for (edge, edge_ind) in zip(self.original_task_graph.edges, range(len(self.original_task_graph.edges))):
            out_node_pos = node_pos[edge[0]]
            in_node_pos = node_pos[edge[1]]
            middle_pos = (np.array(in_node_pos) - np.array(out_node_pos)) / 2 + np.array(out_node_pos) - [0.05, 0]
            ax.text(middle_pos[0], middle_pos[1], str(flow_solution[edge_ind]), fontsize=6)
            # import pdb; pdb.set_trace()
        plt.savefig(graph_img_file)
        plt.clf()
        return node_pos

    def draw_new_taskgraph(self, new_task_graph, new_to_old_node_mapping, new_flow_solution):
        total_num_nodes = self.original_num_tasks + new_task_graph.source_node_info_dict['num_source_nodes'] - 1
        new_pos_dict = {}
        for key in self.nodepos_dict.keys():
            if key == 0:
                new_key = 0
            elif key not in new_to_old_node_mapping:
                new_key = -key
            else:
                new_key = new_to_old_node_mapping.index(key)
            new_pos_dict[new_key] = self.nodepos_dict[key]

        # for new_source in range(1, new_task_graph.source_node_info_dict['num_source_nodes']):

        edges = copy.deepcopy(new_task_graph.edges)
        completed_edges = []
        # add in edges from completed tasks
        for task in [i for i in range(1, self.original_num_tasks) if self.task_completed[i]]:
            old_edges = [edge for edge in self.original_task_graph.edges if edge[0] == task]
            for edge in old_edges:
                if self.task_completed[edge[1]]:
                    new_out_node = -edge[1]
                else:
                    new_out_node = new_to_old_node_mapping.index(edge[1])
                #print(f"appending new edge: ({-edge[0]},{new_out_node})")
                edges.append([-edge[0], new_out_node])
                completed_edges.append(([-edge[0], new_out_node], self.original_task_graph.edges.index(edge)))

        graph = nx.DiGraph()
        graph.add_edges_from(edges)

        label_dict = {}
        new_src_cnt = 0
        for node in graph.nodes:
            label_dict[node] = str(node)
            if node not in new_pos_dict.keys():
                new_pos_dict[node] = (np.random.rand() - 5, new_src_cnt * 2)
                new_src_cnt += 1
        colors = ['b' for _ in range(total_num_nodes)]
        for (node, node_id) in zip(graph.nodes, range(total_num_nodes)):
            if node < 0:
                colors[node_id] = 'k'
            if node in list(range(new_task_graph.source_node_info_dict['num_source_nodes'])):
                colors[node_id] = 'r'
            if node in new_task_graph.source_node_info_dict['in_progress']:
                colors[node_id] = 'y'
            if node == 0:
                colors[node_id] = 'g'

        nodewise_rewards = new_task_graph.reward_model._nodewise_optim_cost_function(new_flow_solution)
        print(f"total reward graphed: {-np.sum(nodewise_rewards)}")
        for i in range(len(nodewise_rewards)):
            label_dict[i] = label_dict[i] + "\n" + "{:.2f}".format(-nodewise_rewards[i])

        fig, ax = plt.subplots()
        graph_img_file = f'real_time_debug/new_graph_{self.current_step}.jpg'
        nx.draw(graph, labels=label_dict, pos=new_pos_dict, ax=ax, node_color=colors, alpha=0.4, node_size=1200)

        for (edge, edge_ind) in zip(new_task_graph.edges, range(len(new_task_graph.edges))):
            out_node_pos = new_pos_dict[edge[0]]
            in_node_pos = new_pos_dict[edge[1]]
            middle_pos = (np.array(in_node_pos) + 3*np.array(out_node_pos)) / 4 - [0.05, 0]
            ax.text(middle_pos[0], middle_pos[1], str(new_flow_solution[edge_ind]), fontsize=6)

        for edge_tuple in completed_edges:
            edge = edge_tuple[0]
            edge_ind = edge_tuple[1]
            out_node_pos = new_pos_dict[edge[0]]
            in_node_pos = new_pos_dict[edge[1]]
            middle_pos = (np.array(in_node_pos) + 3*np.array(out_node_pos)) / 4 - [0.05, 0]
            ax.text(middle_pos[0], middle_pos[1], str(self.current_solution[edge_ind]), fontsize=6)

        plt.savefig(graph_img_file)
        plt.clf()
