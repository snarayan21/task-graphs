from taskgraph import TaskGraph
import networkx as nx
import numpy as np

class RealTimeSolver():

    def __init__(self, taskgraph_arg_dict):
        # initialize task graph with arguments
        # keep track of free and busy agents, completed and incomplete tasks, reward
        # model, etc
        self.original_task_graph = TaskGraph(**taskgraph_arg_dict)
        self.original_num_tasks = self.original_task_graph.num_tasks
        self.original_num_robots = self.original_task_graph.num_robots
        self.task_completed = [False for _ in range(self.original_num_tasks)]
        self.agent_functioning = [True for _ in range(self.original_num_robots)]
        self.agent_free = [True for _ in range(self.original_num_robots)]
        self.current_time = 0.0
        self.current_step = 0

        # solve task graph
        # TODO: will we want to just do one solution at a time? probably
        self.original_task_graph.solve_graph_scipy()
        # save pruned rounded NLP solution as current solution -- set of flows over edges
        self.current_solution = self.original_task_graph.pruned_rounded_baseline_solution
        self.agent_assignment, start_times, finish_times = self.get_assignment(self.original_task_graph, self.current_solution)
        self.ordered_finish_times = np.sort(finish_times)
        fin_time_inds = np.argsort(finish_times)
        self.ordered_finish_times_inds = fin_time_inds[self.ordered_finish_times>0]
        self.ordered_finish_times = self.ordered_finish_times[self.ordered_finish_times>0]
        print(self.agent_assignment)
        print(start_times)
        print(finish_times)
        print(self.ordered_finish_times)
        print(self.ordered_finish_times_inds)

        self.tasks_completed = []

    def step(self, completed_tasks, rewards, free_agents, time):
        # takes in an update from graph manager node on current graph status.
        # uses update to generate a new task allocation plan
        # NOTE: best to keep everything in terms of the original task graph
        # -- translate to new graph, solve, and then immediately translate back
        print(f"Completed tasks: {completed_tasks}")
        print(f"Completed task rewards: {rewards}")
        print(f"Free agents: {free_agents}")
        print(f"Current time: {time}")
        # need to create an update_reward_model function that plugs in completed task rewards
        pass

    def sim_step(self):
        # simulates a step forward in time, calls step function with updated graph
        # will be used to implement disturbances into the simulator to test disturbance
        # response/ robustness in stripped down simulator

        task_completed = self.ordered_finish_times_inds[self.current_step]
        time = self.ordered_finish_times[self.current_step]
        # for now, get exact expected reward
        all_rewards = -self.original_task_graph.reward_model._nodewise_optim_cost_function(self.original_task_graph.pruned_rounded_baseline_solution)
        reward = all_rewards[task_completed]
        free_agents = self.agent_assignment[task_completed]
        self.current_step += 1

        self.step([task_completed],[reward], free_agents, time)

        # need list of tasks completed, list of rewards of those tasks, list of free agents
        # that have just finished completed tasks


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

