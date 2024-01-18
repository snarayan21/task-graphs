from taskgraph import TaskGraph
import networkx as nx
import numpy as np
import copy
import toml

class RealTimeSolver():

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
        self.agent_assignment, self.start_times, self.finish_times = self.get_assignment(self.original_task_graph, self.current_solution)
        self.ordered_finish_times = np.sort(self.finish_times)
        fin_time_inds = np.argsort(self.finish_times)
        self.ordered_finish_times_inds = fin_time_inds[self.ordered_finish_times>0]
        self.ordered_finish_times = self.ordered_finish_times[self.ordered_finish_times>0]
        print(self.agent_assignment)
        print(self.start_times)
        print(self.ordered_finish_times)
        print(self.ordered_finish_times_inds)


    def step(self, completed_tasks, inprogress_tasks, inprogress_task_times, rewards, free_agents, time):
        # inputs:
        # completed_tasks -- list of task IDs of completed tasks
        # inprogress_tasks -- list of task IDs of tasks currently in progress
        # inprogress_task_times -- list of time already spent (s) working on each inprogress task. order same as inprogress_tasks
        # TODO do we need input of agents working on each of these in progress tasks? or can we get that from initial assignment??
        # rewards -- list of rewards from completed tasks, same order as completed tasks
        # free_agents -- list of agent IDs of free agents that have just completed the tasks
        # time -- float current time

        # takes in an update from graph manager node on current graph status.
        # uses update to generate a new task allocation plan
        # NOTE: best to keep everything in terms of the original task graph
        # -- translate to new graph, solve, and then immediately translate back
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


        self.current_time = time

        new_task_graph = self.create_new_taskgraph(inprogress_tasks, inprogress_task_times)

        # solve flow problem
        new_task_graph.solve_graph_scipy()

        # new flow solution
        new_flow_solution = new_task_graph.pruned_rounded_baseline_solution
        import pdb; pdb.set_trace()

        # translate new flow solution to original task graph and save
        # NOTE: after this, the new flow solution may not be valid under flow constraints


        # need to create an update_reward_model function that plugs in completed task rewards
        pass

    def sim_step(self):
        # simulates a step forward in time, calls step function with updated graph
        # will be used to implement disturbances into the simulator to test disturbance
        # response/ robustness in stripped down simulator

        task_completed = self.ordered_finish_times_inds[self.current_step]
        time = self.ordered_finish_times[self.current_step]
        inprogress_tasks = []
        inprogress_task_times = []
        for task in range(self.original_num_tasks):
            if time > self.start_times[task] and time < self.finish_times[task] and task != task_completed:
                inprogress_tasks.append(task)
                inprogress_task_times.append(time-self.start_times[task])
        # for now, get exact expected reward
        all_rewards = -self.original_task_graph.reward_model._nodewise_optim_cost_function(self.original_task_graph.pruned_rounded_baseline_solution)
        reward = all_rewards[task_completed]
        free_agents = self.agent_assignment[task_completed]
        self.current_step += 1

        self.step([task_completed], inprogress_tasks, inprogress_task_times, [reward], free_agents, time)

        # need list of tasks completed, list of rewards of those tasks, list of free agents
        # that have just finished completed tasks

    def create_new_taskgraph(self, inprogress_tasks, inprogress_task_times):
        # creates a new task graph from old task graph parameters
        # revises reward model, replacing completed tasks with constant reward values in the influence
        # functions of future tasks, and replacing in-progress tasks with expected reward values in the
        # same manner
        # removes completed and in-progress tasks from the graph
        # maintains a map from new nodes/edges to old
        # SIDE NOTE how do we specify multiple source nodes? have to change scipy constraints

        # create copy of original task graph args
        new_args = copy.deepcopy(self.original_args)
        new_args['edges'] = self.original_task_graph.reward_model.edges # fix any ordering discrepancies


        edges_to_delete_inds = [] # each list entry i is the id of an edge that we need to delete
        ghost_node_param_dict = {}
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

                # TODO this only works if each outgoing neighbor has only one incoming node that gets completed this step --
                #  this is probably a valid assumption most of the time? Will only be violated if we want to do batch processing of steps
                outgoing_neighbors = [outgoing_edges[i][1] for i in range(len(outgoing_edges))]
                for (nbr, nbr_edge) in zip(outgoing_neighbors, outgoing_edge_inds):
                    #tuple of (influence_type, influence_params, reward)
                    ghost_node_param_dict[nbr] = (self.original_args['dependency_types'][nbr_edge],
                                                   self.original_args['dependency_params'][nbr_edge],
                                                   self.task_rewards[task])

        param_dict_string_keys = copy.deepcopy(ghost_node_param_dict)
        for key in ghost_node_param_dict.keys():
            param_dict_string_keys[str(key)] = param_dict_string_keys.pop(key)
        new_args['ghost_node_param_dict'] = param_dict_string_keys

        edges_to_delete_inds.sort(reverse=True) # sort edge ids from highest to lowest to delete them
        for edge_id in edges_to_delete_inds: # delete edges from taskgraph args
            new_args['edges'].pop(edge_id)
            new_args['dependency_params'].pop(edge_id)
            new_args['dependency_types'].pop(edge_id)

        # how many new nodes total?
        num_new_sources = 1 + len(inprogress_tasks)

        # add new source node that connects to nodes without incoming edges that are NOT in progress
        nodes_to_keep = [task for task in range(self.original_num_tasks) if not self.task_completed[task]]
        nodes_to_connect_to_source = []
        for task in nodes_to_keep:
            node_has_in_neighbors = False
            if not task in inprogress_tasks:
                for edge_id in new_args['edges']: #iterate through edges we're keeping (w/ original names)
                    if task == edge_id[1]:
                        node_has_in_neighbors = True

            if not node_has_in_neighbors:
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
        new_to_old_node_mapping = [-1]*num_new_sources + [task for task in range(self.original_num_tasks) if not self.task_completed[task]]


        # add source nodes for each in-progress task
        # add number of agents at each source node that are working on the corresponding task




        # rename old edges
        for edge_id in range(len(new_args['edges'])):
            new_args['edges'][edge_id] = [new_to_old_node_mapping.index(new_args['edges'][edge_id][0]),
                                          new_to_old_node_mapping.index(new_args['edges'][edge_id][1])]

        # add new edges
        cur_source = 1 # start on 1, leave 0 for new source node for free agents
        for node in inprogress_tasks:
            new_node_name = new_to_old_node_mapping.index(node)
            new_args['edges'] = [[cur_source, new_node_name]] + new_args['edges']
            new_args['dependency_types'] = ['polynomial'] + new_args['dependency_types']
            new_args['dependency_params'] = [[0.,0.,0.]] + new_args['dependency_params']
            cur_source += 1

        for node in nodes_to_connect_to_source:
            if node not in inprogress_tasks:
                new_node_name = new_to_old_node_mapping.index(node)
                new_args['edges'] = [[0,new_node_name]] + new_args['edges']
                new_args['dependency_types'] = ['polynomial'] + new_args['dependency_types']
                new_args['dependency_params'] = [[0.,0.,0.]] + new_args['dependency_params']


        new_edges_old_names = [[new_to_old_node_mapping[e[0]],new_to_old_node_mapping[e[1]]] for e in new_args['edges']]

        source_neighbors = nodes_to_connect_to_source + [new_to_old_node_mapping.index(t) for t in inprogress_tasks]

        # remove tasks from taskgraph
        new_args['coalition_params'] = [[0., 0., 0.]]*num_new_sources + [new_args['coalition_params'][i] for i in range(self.original_num_tasks) if not self.task_completed[i]]
        new_args['coalition_types'] = ['null']*num_new_sources + [new_args['coalition_types'][i] for i in range(self.original_num_tasks) if not self.task_completed[i]]
        # TODO have to make sure that rewards aren't artificially inflated or zeroed out by new source nodes
        new_args['aggs'] = ['or']*num_new_sources + [new_args['aggs'][i] for i in range(self.original_num_tasks) if not self.task_completed[i]]
        if new_args['nodewise_coalition_influence_agg_list'] is not None:
            new_args['nodewise_coalition_influence_agg_list'] = ['sum']*num_new_sources + [new_args['nodewise_coalition_influence_agg_list'][i] for i in range(self.original_num_tasks) if not self.task_completed[i]]
        for task in source_neighbors:
            if new_args['nodewise_coalition_influence_agg_list'] is not None:
                new_args['nodewise_coalition_influence_agg_list'][task] = 'sum'

        new_args['task_times'] = [0.]*num_new_sources + [self.original_task_graph.task_times[i] for i in range(self.original_num_tasks) if not self.task_completed[i]]
        # adjust in-progress task durations to reflect current time
        for task in inprogress_tasks:
            new_node_name = new_to_old_node_mapping.index(task)
            new_args['task_times'][new_node_name] = new_args['task_times'][new_node_name] - inprogress_task_times[inprogress_tasks.index(task)]

        new_args['num_tasks'] = len(new_to_old_node_mapping)

        source_node_info_dict = {'num_source_nodes': num_new_sources}
        new_args['source_node_info_dict'] = source_node_info_dict

        args_file = 'real_time_debug/new_args.toml'
        with open(args_file, "w") as f:
                toml.dump(new_args,f)

        # create new taskgraph with args
        new_task_graph = TaskGraph(**new_args)

        return new_task_graph

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
            print(f"PROCESSING NODE {current_node}")
            incoming_edges = [list(e) for e in taskgraph.task_graph.in_edges(current_node)]
            incoming_edge_inds = [taskgraph.reward_model.edges.index(e) for e in incoming_edges]
            print(f"incoming edges: {incoming_edges}")
            if len(incoming_edges) > 0:
                # if no incoming flow, no assignment, task time of 0
                if np.array([flow[incoming_edge_inds[i]]<=0.000001 for i in range(len(incoming_edges))]).all():
                    task_start_times[int(current_node)] = 0.0
                    incomplete_nodes.append(current_node)
                    print("no incoming flow: task is not completed")
                else:
                    task_start_times[int(current_node)] = max([task_finish_times[int(incoming_edges[i][0])] for i in range(len(incoming_edges)) if not incoming_edges[i][0] in np.array(incomplete_nodes)])
                    for ind in incoming_edge_inds:
                        num_agents = int(round(flow[ind]*taskgraph.num_robots))
                        edge = taskgraph.reward_model.edges[ind]
                        print(f"{num_agents} flowing from edge {edge}")
                        incoming_node = edge[0]
                        # TODO make the following line POP rather than just grab
                        agent_list = []
                        for _ in range(num_agents):
                            agent_list.append(temp_agent_assignments[incoming_node].pop(0))
                        temp_agent_assignments[current_node] += agent_list
                        agent_assignments[current_node] += agent_list

            else:
                print("at start node")
                task_start_times[int(current_node)] = 0
                agent_assignments[int(current_node)] = [i for i in range(taskgraph.num_robots)]
                temp_agent_assignments[int(current_node)] = [i for i in range(taskgraph.num_robots)]
            task_finish_times[int(current_node)] = task_start_times[int(current_node)] + taskgraph.task_times[int(current_node)]

        for node in incomplete_nodes:
            task_finish_times[int(node)] = 0.0
        return agent_assignments, task_start_times, task_finish_times

