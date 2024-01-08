from taskgraph import TaskGraph
import networkx as nx
import numpy as np
import copy

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


    def step(self, completed_tasks, inprogress_tasks, rewards, free_agents, time):
        # inputs:
        # completed_tasks -- list of task IDs of completed tasks
        # inprogress_tasks -- list of task IDs of tasks currently in progress
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

        self.current_time = time



        # need to create an update_reward_model function that plugs in completed task rewards
        pass

    def sim_step(self):
        # simulates a step forward in time, calls step function with updated graph
        # will be used to implement disturbances into the simulator to test disturbance
        # response/ robustness in stripped down simulator

        task_completed = self.ordered_finish_times_inds[self.current_step]
        time = self.ordered_finish_times[self.current_step]
        inprogress_tasks = []
        for task in range(self.original_num_tasks):
            if time > self.start_times[task] and time < self.finish_times[task] and task != task_completed:
                inprogress_tasks.append(task)
        # for now, get exact expected reward
        all_rewards = -self.original_task_graph.reward_model._nodewise_optim_cost_function(self.original_task_graph.pruned_rounded_baseline_solution)
        reward = all_rewards[task_completed]
        free_agents = self.agent_assignment[task_completed]
        self.current_step += 1

        self.step([task_completed], inprogress_tasks, [reward], free_agents, time)

        # need list of tasks completed, list of rewards of those tasks, list of free agents
        # that have just finished completed tasks

    def create_new_taskgraph(self, ):
        # creates a new task graph from old task graph parameters
        # revises reward model, replacing completed tasks with constant reward values in the influence
        # functions of future tasks, and replacing in-progress tasks with expected reward values in the
        # same manner
        # removes completed and in-progress tasks from the graph
        # maintains a map from new nodes/edges to old
        # SIDE NOTE how do we specify multiple source nodes? have to change scipy constraints

        # create copy of original task graph args
        new_args = copy.deepcopy(self.original_args)


        edges_to_delete = [] # each list entry i is the id of an edge that we need to delete
        ghost_node_params_dict = {}
        for task in range(self.original_num_tasks):
            if self.task_completed[task]:
                # create list of completed task outgoing edge inds
                outgoing_edges = [list(e) for e in self.original_task_graph.task_graph.out_edges(task)]
                edges_to_delete = edges_to_delete + outgoing_edges

                outgoing_edge_inds = [self.original_task_graph.reward_model.edges.index(e) for e in outgoing_edges]
                # TODO this only works if each outgoing neighbor has only one incoming node that gets completed this step --
                #  this is probably a valid assumption most of the time? Will only be violated if we want to do batch processing of steps
                outgoing_neighbors = [outgoing_edges[i][1] for i in range(len(outgoing_edges))]
                for (nbr, nbr_edge) in zip(outgoing_neighbors, outgoing_edge_inds):
                    #tuple of (influence_type, influence_params, reward)
                    ghost_node_params_dict[nbr] = (self.original_args['dependency_types'][nbr_edge],
                                                   self.original_args['dependency_params'][nbr_edge],
                                                   self.task_rewards[task])

        # delete edges from taskgraph args
        for edge in edges_to_delete.sort(reverse=True): # sort edge ids from highest to lowest to delete them
            new_args['edges'].pop(edge)
            new_args['dependency_params'].pop(edge)
            new_args['dependency_types'].pop(edge)

        # add new source nodes where nodes have no incoming edges
        nodes_to_keep = [task for task in range(self.original_num_tasks) if not self.task_completed[task]]
        nodes_without_in_neighbors = []
        for task in nodes_to_keep:
            node_has_in_neighbors = False
            for edge in new_args['edges']: #iterate through edges we're keeping (w/ original names)
                if task == edge[1]:
                    node_has_in_neighbors = True

            if not node_has_in_neighbors:
                nodes_without_in_neighbors.append(task)

        # add ONE source node that connects to all nodes without incoming neighbors
        # TODO: may want to generalize this to several new source nodes w/ different agent capacities on each
        # generate mapping of new taskgraph node ids to old taskgraph node ids
        # entry of -1 corresponds to NEW node -- no corresponding node on old graph
        new_to_old_node_mapping = [-1] + [task for task in range(self.original_num_tasks) if not self.task_completed[task]]

        # rename old edges
        for edge_id in range(len(new_args['edges'])):
            new_args['edges'][edge_id] = [new_to_old_node_mapping.index(new_args['edges'][edge_id][0]),
                                          new_to_old_node_mapping.index(new_args['edges'][edge_id][1])]

        # add new edges
        for node in nodes_without_in_neighbors:
            new_node_name = new_to_old_node_mapping.index(node)
            new_args['edges'] = [[0,new_node_name]] + new_args['edges']


        # remove tasks from taskgraph
        new_args['coalition_params'] = [new_args['coalition_params'][i] for i in range(self.original_num_tasks) if not self.task_completed[i]]
        new_args['coalition_types'] = [new_args['coalition_types'][i] for i in range(self.original_num_tasks) if not self.task_completed[i]]
        new_args['aggs'] = [new_args['aggs'][i] for i in range(self.original_num_tasks) if not self.task_completed[i]]
        if new_args['nodewise_coalition_influence_agg_list'] is not None:
            new_args['nodewise_coalition_influence_agg_list'] = [new_args['nodewise_coalition_influence_agg_list'][i] for i in range(self.original_num_tasks) if not self.task_completed[i]]
        new_args['task_times'] = [new_args['task_times'][i] for i in range(self.original_num_tasks) if not self.task_completed[i]]

        # create new taskgraph with args
        new_task_graph = TaskGraph(**new_args)

        # solve flow problem
        new_task_graph.solve_graph_scipy()

        # new flow solution
        new_flow_solution = new_task_graph.pruned_rounded_baseline_solution


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

