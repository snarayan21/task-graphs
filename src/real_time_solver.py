from taskgraph import TaskGraph
import networkx as nx
import numpy as np
import copy
import toml
import matplotlib.pyplot as plt


class RealTimeSolver:

    def __init__(self, taskgraph_arg_dict, trial_dir='real_time_debug', draw_graph=True):
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
        self.trial_dir = trial_dir

        # solve task graph
        self.original_task_graph.solve_graph_scipy()
        # save pruned rounded NLP solution as current solution -- set of flows over edges
        self.current_solution = self.original_task_graph.pruned_rounded_baseline_solution
        print(f"Original pruned rounded solution: {self.current_solution}")
        self.current_nodewise_rewards = -self.original_task_graph.reward_model._nodewise_optim_cost_function(
            self.current_solution, debug=False)
        self.original_agent_assignment, self.original_start_times, self.original_finish_times = self.get_assignment(self.original_task_graph, self.current_solution, list(range(self.original_num_tasks)))
        # TODO test ITT here -- not working
        #import pdb; pdb.set_trace()
        for task in range(self.original_num_tasks):
            if self.original_finish_times[task] > self.original_task_graph.makespan_constraint:
                self.current_nodewise_rewards[task] = 0
        # TODO SMALL BUG: why is original solution resulting in not overrunning makespan constraint but updated solutions are???
        self.original_solution = copy.deepcopy(self.current_solution)
        self.original_rewards = copy.deepcopy(self.current_nodewise_rewards)


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

        if draw_graph:
            self.nodepos_dict = self.draw_original_taskgraph(self.current_solution, self.trial_dir)

        self.oracle_reward_model = None

        self.all_taskgraphs = [self.original_task_graph]
        self.all_solutions = [copy.deepcopy(self.current_solution)]
        self.all_node_mappings = [list(range(self.original_num_tasks))]
        self.step_times = [0.0]
        self.task_done_history = [[0]]
        self.task_assignment_history = [copy.deepcopy(self.current_agent_assignment)]
        self.catastrophic_perturbations = []

    def step(self, completed_tasks, inprogress_tasks, inprogress_task_times, inprogress_task_coalitions, rewards,
             free_agents, time, draw_graph):
        # inputs:
        # completed_tasks -- list of task IDs of completed tasks
        # inprogress_tasks -- list of task IDs of tasks currently in progress
        # inprogress_task_times -- list of time already spent (s) working on each inprogress task. order same as inprogress_tasks
        # inprogress_task_coalitions -- list of lists, where list i is the set of agents currently working on inprogress task i (same order as inprogress_tasks)
        # rewards -- list of rewards from completed tasks, same order as completed tasks
        # free_agents -- list of agent IDs of free agents that have just completed the tasks
        # time -- float current time
        # draw_graph -- boolean plot and save to file graph images - use for testing, too slow for full experiments

        # takes in an update from graph manager node on current graph status.
        # uses update to generate a new task allocation plan
        # NOTE: best to keep everything in terms of the original task graph
        # -- translate to new graph, solve, and then immediately translate back
        print(f"-------------------NEW STEP {self.current_step} ----------------------")
        print(f"Completed tasks: {completed_tasks}")
        print(f"Completed task rewards: {rewards}")
        print(f"Inprogress tasks: {inprogress_tasks}")
        print(f"Free agents: {free_agents}")
        print(f"Current time: {time}")

        self.step_times.append(time)
        self.task_done_history.append([])
        task_it = 0
        for task in completed_tasks:
            self.task_completed[task] = True
            self.task_rewards[task] = rewards[task_it]
            self.task_done_history[self.current_step].append(task)
            task_it += 1

        # initial source node is automatically always completed when step is called
        self.task_completed[0] = True
        self.task_rewards[0] = 0

        new_completed_task = True
        all_completed_tasks = copy.deepcopy(completed_tasks)
        while new_completed_task:
            new_completed_task = False
            # mark tasks with incoming edges to the completed task complete
            completed_tasks_this_it = []
            for task in all_completed_tasks:
                if task != 0:
                    in_edges = [e for e in self.original_task_graph.edges if e[1] == task]
                    for edge in in_edges:
                        if not self.task_completed[edge[0]]:
                            self.task_completed[edge[0]] = True
                            new_completed_task = True
                            completed_tasks_this_it.append(edge[0])
                            self.task_done_history[self.current_step].append(edge[0])
                            if edge[0] in inprogress_tasks:
                                node_to_remove_ind = inprogress_tasks.index(edge[0])
                                inprogress_tasks.pop(node_to_remove_ind)
                                inprogress_task_coalitions.pop(node_to_remove_ind)
                                inprogress_task_times.pop(node_to_remove_ind)
                            # if self.current_finish_times[edge[0]] < self.original_task_graph.makespan_constraint:
                            #     self.task_rewards[edge[0]] = self.current_actual_and_expected_rewards[edge[0]]
                            # else:
                            #    self.task_rewards[edge[0]] = 0.0
                            # if task was not completed, it yields zero reward
                            self.task_rewards[edge[0]] = 0.0
                            self.current_start_times[edge[0]] = 0.0
                            self.current_finish_times[edge[0]] = 0.0

                            print(f"PRECEDING TASK {edge[0]} COMPLETED WITH REWARD {self.task_rewards[edge[0]]}, MARKING TASK {edge[0]} AS COMPLETE.")
            all_completed_tasks = all_completed_tasks + completed_tasks_this_it

        new_completed_task = True
        all_completed_inprogress = copy.deepcopy(inprogress_tasks)
        while new_completed_task:
            new_completed_task = False
            newly_added = []
            for task in inprogress_tasks:
                in_edges = [e for e in self.original_task_graph.edges if e[1] == task]
                for edge in in_edges:
                    if not self.task_completed[edge[0]]:
                        self.task_completed[edge[0]] = True
                        new_completed_task = True
                        self.task_done_history[self.current_step].append(edge[0])
                        self.task_rewards[edge[0]] = 0.0
                        newly_added.append(edge[0])
            all_completed_inprogress = all_completed_inprogress + newly_added


        if len(self.task_completed) - sum(self.task_completed) < 2:
            print(f"After task removal, {len(self.task_completed) - sum(self.task_completed)} tasks remain.")
            if not np.all(self.task_completed):
                for task in range(self.original_num_tasks):
                    if not self.task_completed[task]:
                        if self.current_finish_times[task] < self.original_task_graph.makespan_constraint:
                            if len(self.current_agent_assignment[task]) > 0:
                                self.task_rewards[task] = self.current_actual_and_expected_rewards[task]
                            else:
                                self.task_rewards[task] = 0.0
                            self.task_done_history[self.current_step].append(task)
                            print(f"Adding {self.task_rewards[task]} reward for last remaining task {task}, which will be completed in time ")
            print("TERMINATING REAL TIME SOLVER: RETURNING TRUE FOR STEP")
            self.task_assignment_history.append(copy.deepcopy(self.current_agent_assignment))
            return True
        self.current_time = time

        new_task_graph, new_to_old_node_mapping = self.create_new_taskgraph(inprogress_tasks, inprogress_task_times,
                                                                            inprogress_task_coalitions)

        # solve flow problem
        new_task_graph.solve_graph_scipy()

        # new flow solution
        raw_new_flow_solution = new_task_graph.pruned_rounded_baseline_solution
        print(f"Initial flow solution: {raw_new_flow_solution}")
        self.all_taskgraphs.append(new_task_graph)
        self.all_node_mappings.append(new_to_old_node_mapping)
        new_flow_solution, inprogress_coalitions_source_it = self.check_and_update_solution(raw_new_flow_solution)

        new_agent_assignment, new_start_times, new_finish_times = self.get_assignment(new_task_graph, new_flow_solution, new_to_old_node_mapping, inprogress_coalitions_source_it=inprogress_coalitions_source_it)

        print(f"new start times on new graph: {new_start_times}")
        print(f"new finish times on new graph: {new_finish_times}")

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
                    if new_flow_solution_old_edges[old_graph_in_edge_inds[0]] > 1:
                        #import pdb; pdb.set_trace()
                        raise Exception
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
                if new_finish_times[new_task_name] + self.current_time < self.original_task_graph.makespan_constraint:
                    current_actual_and_predicted.append(new_flow_reward_nodewise_new_graph[new_task_name])
                else:
                    current_actual_and_predicted.append(0.0)
                    print(f"OG task {task} not completed in time: {new_flow_reward_nodewise_new_graph[new_task_name]} reward not added to current_acutal_and_predicted")
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


        if draw_graph:
            self.draw_new_taskgraph(new_task_graph, new_to_old_node_mapping, new_flow_solution, graph_img_dir=self.trial_dir)
        self.current_solution = new_flow_solution_old_edges

        # redundant?? unneeded --> new_agent_assignment, new_start_times, new_finish_times = self.get_assignment(new_task_graph, new_flow_solution)
        for i in range(len(new_agent_assignment)):
            old_node_name = new_to_old_node_mapping[i]
            self.current_agent_assignment[old_node_name] = new_agent_assignment[i]
            if new_finish_times[i] != 0.0:
                self.current_start_times[old_node_name] = new_start_times[i] + self.current_time
                self.current_finish_times[old_node_name] = new_finish_times[i] + self.current_time
            else: # else fin time of zero indicates task is not assigned this iteration
                self.current_finish_times[old_node_name] = 0.0
                self.current_start_times[old_node_name] = 0.0

        for i in range(self.original_num_tasks):
            if len(self.current_agent_assignment[i]) == 0 and self.current_finish_times[i] > 0:
                #import pdb; pdb.set_trace()
                raise Exception
        self.current_ordered_finish_times = np.sort(self.current_finish_times)
        fin_time_inds = np.argsort(self.current_finish_times)
        self.current_ordered_finish_times_inds = fin_time_inds[self.current_ordered_finish_times > self.current_time]
        self.current_ordered_finish_times = self.current_ordered_finish_times[self.current_ordered_finish_times > self.current_time]
        print(f"new agent assignments: {self.current_agent_assignment}")
        print(f"new start times: {self.current_start_times}")
        print(f"new finish times: {self.current_finish_times}")
        print(f"new task order: {self.current_ordered_finish_times_inds}")
        self.current_actual_and_expected_rewards = current_actual_and_predicted
        self.all_solutions.append(new_flow_solution)
        self.task_assignment_history.append(copy.deepcopy(self.current_agent_assignment))
        if np.all(new_flow_solution == 0.):
            for task in [t for t in range(self.original_num_tasks) if not self.task_completed[t]]:
                new_task_ind = new_to_old_node_mapping.index(task)
                if self.current_finish_times[task] < self.original_task_graph.makespan_constraint:
                    self.task_rewards[task] = new_flow_reward_nodewise_new_graph[new_task_ind]
            return True  # FINISHED
        return False  # NOT YET FINISHED


    def sim_step(self, perturbation=None, perturbation_params=None, perturb_model_params=None, draw_graph=True):
        # simulates a step forward in time, calls step function with updated graph
        # will be used to implement disturbances into the simulator to test disturbance
        # response/ robustness in stripped down simulator
        for task in self.current_ordered_finish_times_inds:
            if not self.task_completed[task]:
                task_completed = task
                time = self.current_finish_times[task]
                break

        if len(self.current_ordered_finish_times_inds)==0 or len(self.all_taskgraphs[self.current_step].edges)==1:
            for task in [t for t in range(self.original_num_tasks) if not self.task_completed[t]]:
                if self.current_finish_times[task] < self.original_task_graph.makespan_constraint:
                    self.task_rewards[task] = self.current_actual_and_expected_rewards[task]
            #if len(self.all_taskgraphs[self.current_step].edges)==1 and len(self.current_ordered_finish_times_inds) != 0 :
            #    self.task_rewards[task_completed] = self.current_actual_and_expected_rewards[task_completed]
            print("REAL TIME REALLOCATION COMPLETE: NO FURTHER TASKS")
            return True

        self.current_step += 1

        inprogress_tasks = []
        inprogress_task_times = []
        inprogress_coalitions = []
        for task in range(self.original_num_tasks):
            if time > self.current_start_times[task] and time < self.current_finish_times[task] and task != task_completed:
                inprogress_tasks.append(task)
                # calculate the time we've worked on the task since starting it or the last step (in the case where
                # tasks take two steps), whichever is more recent
                inprogress_task_times.append(time - max(self.current_start_times[task],
                                                        self.step_times[self.current_step-1]))
                inprogress_coalitions.append(self.current_agent_assignment[task])
        if perturb_model_params is None:
            if perturbation is None:
                # if no perturbation, get exact expected reward
                reward = self.current_actual_and_expected_rewards[task_completed]
                expected_reward = reward
            elif perturbation == 'gaussian':
                mean = self.current_actual_and_expected_rewards[task_completed]
                std = perturbation_params*self.current_actual_and_expected_rewards[task_completed]
                reward = max(0, np.random.normal(loc=mean, scale=std)) # ensure positive reward
                expected_reward = copy.deepcopy(self.current_actual_and_expected_rewards[task_completed])
            elif perturbation == 'catastrophic':
                if np.random.rand() < perturbation_params:
                    reward = 0.0
                    self.catastrophic_perturbations.append(task_completed)
                else:
                    reward = self.current_actual_and_expected_rewards[task_completed]
                expected_reward = copy.deepcopy(self.current_actual_and_expected_rewards[task_completed])
        else:
            if perturbation is None:
                # if no perturbation, get exact expected reward FROM ORACLE MODEL
                # TODO how to do this??
                reward = self.current_actual_and_expected_rewards[task_completed]
                expected_reward = reward
            elif perturbation == 'gaussian':
                # TODO
                mean = self.current_actual_and_expected_rewards[task_completed]
                std = perturbation_params*self.current_actual_and_expected_rewards[task_completed]
                reward = max(0, np.random.normal(loc=mean, scale=std)) # ensure positive reward
                expected_reward = copy.deepcopy(self.current_actual_and_expected_rewards[task_completed])
            elif perturbation == 'catastrophic':
                if np.random.rand() < perturbation_params:
                    reward = 0.0
                else:
                    reward = self.current_actual_and_expected_rewards[task_completed]
                expected_reward = copy.deepcopy(self.current_actual_and_expected_rewards[task_completed])

        free_agents = self.current_agent_assignment[task_completed]
        mission_complete = self.step([task_completed], inprogress_tasks, inprogress_task_times, inprogress_coalitions,
                                     [reward], free_agents, time, draw_graph=draw_graph)
        print(f"original makespan constraint: {self.original_task_graph.makespan_constraint}")
        remaining_tasks = [i for i in range(self.original_num_tasks) if not self.task_completed[i]]
        remaining_task_times = [self.original_task_graph.task_times[i] for i in remaining_tasks]
        print(f"Task times remaining: {remaining_task_times} for tasks {remaining_tasks}.")
        if np.all(remaining_task_times > (self.original_task_graph.makespan_constraint-time)):
            print(f"RAN OUT OF TIME: REAL TIME REALLOCATION TERMINATED. Task times remaining: {remaining_task_times} for tasks {remaining_tasks}.")
            print(f"remaining time: {self.original_task_graph.makespan_constraint-time}")
            return True, reward, expected_reward

        return mission_complete, reward, expected_reward# return true if DONE

        # need list of tasks completed, list of rewards of those tasks, list of free agents
        # that have just finished completed tasks
    def sim_step_perturbed_model(self, perturbation=None, perturbation_params=None, perturb_model_params=None, draw_graph=True):
        if self.oracle_reward_model is None:
            self.oracle_reward_model = self.create_oracle_reward_model(sigma=perturb_model_params)

        # call sim_step with some new input -- likely a binary -- that performs a reward sample with the oracle
        # model instead of the solver model
        return self.sim_step(perturbation=perturbation, perturbation_params=perturbation_params, perturb_model_params=perturb_model_params, draw_graph=draw_graph)


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
        if 'run_minlp' in new_args.keys():
            new_args['run_minlp'] = False
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

        # which nodes have an incoming edge from a completed task that we need to consider?
        nodes_with_deleted_incoming_edges = []
        #amount_deleted_incoming_flow = []
        for i in [j for j in range(len(self.task_completed)) if self.task_completed[j]]:
            out_neighbors_i = [edge[1] for edge in self.original_task_graph.edges if edge[0] == i]
            for nbr in out_neighbors_i:
                if (nbr not in inprogress_tasks) and (nbr not in nodes_with_deleted_incoming_edges) and not self.task_completed[nbr]:
                    #if self.current_solution[self.original_task_graph.edges.index((i, nbr))] > 0.00001:
                    nodes_with_deleted_incoming_edges.append(nbr)
                    #amount_deleted_incoming_flow.append(self.current_solution[self.original_task_graph.edges.index((i, nbr))])

        # how many new nodes total
        num_new_sources = 1 + len(inprogress_tasks) #+ len(nodes_with_deleted_incoming_edges)

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
        for node in (inprogress_tasks):  # + nodes_with_deleted_incoming_edges):
            new_node_name = new_to_old_node_mapping.index(node)
            new_args['edges'] = [[cur_source, new_node_name]] + new_args['edges']
            new_args['dependency_types'] = ['polynomial'] + new_args['dependency_types']
            new_args['dependency_params'] = [[0., 0., 0.]] + new_args['dependency_params']
            cur_source += 1

        # if the only nodes to connect to source are inprogress nodes, then connect the source to
        # all outgoing neighbors of inprogress nodes
        if np.all([node in inprogress_tasks for node in (nodes_to_connect_to_source +
                  [node for node in nodes_with_deleted_incoming_edges if node not in nodes_to_connect_to_source])]):
            for task in inprogress_tasks:
                new_task_name = new_to_old_node_mapping.index(task)
                out_nbrs = [e[1] for e in new_args['edges'] if e[0] == new_task_name]
                for out_nbr in out_nbrs:
                    if out_nbr not in nodes_to_connect_to_source:
                        nodes_to_connect_to_source.append(new_to_old_node_mapping[out_nbr])

        nodes_to_connect_to_source_new_names = []
        for node in (nodes_to_connect_to_source + [node for node in nodes_with_deleted_incoming_edges if node not in nodes_to_connect_to_source]):
            if node not in inprogress_tasks:
                new_node_name = new_to_old_node_mapping.index(node)
                nodes_to_connect_to_source_new_names.append(new_node_name)
                new_args['edges'] = [[0, new_node_name]] + new_args['edges']
                new_args['dependency_types'] = ['polynomial'] + new_args['dependency_types']
                new_args['dependency_params'] = [[0., 0., 0.]] + new_args['dependency_params']

        new_edges_old_names = [[new_to_old_node_mapping[e[0]], new_to_old_node_mapping[e[1]]] for e in
                               new_args['edges']]
        new_args['inter_task_travel_times'] = [0.0 for _ in range(len(new_args['edges']))]
        for edge_id in range(len(new_edges_old_names)):
            if new_edges_old_names[edge_id] in self.original_task_graph.edges:
                old_edge_id = self.original_task_graph.edges.index(new_edges_old_names[edge_id])
                new_args['inter_task_travel_times'][edge_id] = self.original_task_graph.inter_task_travel_times[old_edge_id]


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
                        in_edge = in_edge[0]
                        in_edge_ind = new_args['edges'].index(in_edge)
                        new_args['dependency_params'][in_edge_ind] = [1.0, 0.0, 0.0]
                else:
                    new_args['nodewise_coalition_influence_agg_list'][task] = 'sum'

        new_args['task_times'] = [0.] * num_new_sources + [self.original_task_graph.task_times[i] for i in range(self.original_num_tasks) if not self.task_completed[i]]
        # adjust in-progress task durations to reflect current time
        for task in inprogress_tasks:
            new_node_name = new_to_old_node_mapping.index(task)
            old_task_time = self.all_taskgraphs[self.current_step-1].task_times[self.all_node_mappings[self.current_step-1].index(task)]
            new_args['task_times'][new_node_name] = old_task_time - inprogress_task_times[inprogress_tasks.index(task)]
            if new_args['task_times'][new_node_name] < 0:
                #import pdb; pdb.set_trace()
                raise Exception
        new_args['num_tasks'] = len(new_to_old_node_mapping)
        # TODO need to update inter-task travel time
        node_capacities = [0.]
        for coalition in inprogress_task_coalitions:
            node_capacities.append(len(coalition) / self.original_num_robots)
        total_inprogress_coalitions = np.sum(np.array(node_capacities))
        node_capacities[0] = 1 - total_inprogress_coalitions

        source_node_info_dict = {'num_source_nodes': num_new_sources,
                                 'node_capacities': node_capacities,
                                 'in_progress': [new_to_old_node_mapping.index(t) for t in inprogress_tasks]}

        new_args['source_node_info_dict'] = source_node_info_dict
        new_args['makespan_constraint'] = (self.original_task_graph.makespan_constraint - self.current_time)/(np.sum(new_args['task_times']))

        args_file = 'real_time_debug/new_args.toml'
        with open(args_file, "w") as f:
            toml.dump(new_args, f)

        # create new taskgraph with args
        new_task_graph = TaskGraph(**new_args)
        if abs(new_task_graph.makespan_constraint - self.original_task_graph.makespan_constraint + self.current_time) > 0.00001:
            #import pdb; pdb.set_trace()
            raise Exception
        return new_task_graph, new_to_old_node_mapping

    def check_and_update_solution(self, new_flow_solution):
        # check newest solution against prior solutions
        # if appropriate, update newest solution with prior solutions
        new_task_graph = self.all_taskgraphs[self.current_step]
        new_to_old_node_mapping = self.all_node_mappings[self.current_step]
        old_reward_nodewise = new_task_graph.reward_model._nodewise_optim_cost_function(new_flow_solution)
        _, _, old_fin_times = self.get_assignment(new_task_graph, new_flow_solution, new_to_old_node_mapping)
        for task in range(new_task_graph.num_tasks):
            if old_fin_times[task] + self.current_time > self.original_task_graph.makespan_constraint:
                print(f"removing {task} from old solution reward -- finished after deadline")
                old_reward_nodewise[task] = 0
        old_reward = -np.sum(old_reward_nodewise)
        # initialize best reward with current solution's reward
        old_solution_new_graph_rewards = [-10000000 for _ in range(self.current_step)]
        old_solution_new_graph_list = [[] for _ in range(self.current_step)]

        graph_ct = self.current_step-1
        for (last_graph, last_solution, last_node_mapping) in zip(reversed(self.all_taskgraphs[0:self.current_step]),
                                     reversed(self.all_solutions[0:self.current_step]),
                                     reversed(self.all_node_mappings[0:self.current_step])):
            proposed_solution = np.zeros(new_task_graph.num_edges)
            updated_edges = [False for _ in range(new_task_graph.num_edges)]

            # sum up inflow into the tasks on the old graph that are currently in progress on the new graph
            # update edge weights to corresponding source --> inprogress task edges in current graph
            source_ct = 1
            for inprogress_task in new_task_graph.source_node_info_dict['in_progress']:
                og_inprogress_task = new_to_old_node_mapping[inprogress_task]
                last_graph_inprogress_task = last_node_mapping.index(og_inprogress_task)
                last_graph_in_edges = [e for e in last_graph.edges if e[1] == last_graph_inprogress_task]
                last_graph_in_edge_inds = [last_graph.edges.index(e) for e in last_graph_in_edges]
                last_graph_inflow = np.sum([last_solution[i] for i in last_graph_in_edge_inds])
                current_graph_edge_id = new_task_graph.edges.index((source_ct, inprogress_task))
                proposed_solution[current_graph_edge_id] = last_graph_inflow
                print(f"Updated inprogress task {inprogress_task} (old name {og_inprogress_task}: changed edge {(source_ct, inprogress_task)} to {last_graph_inflow}")
                updated_edges[current_graph_edge_id] = True
                source_ct += 1

            source_out_edges = [e for e in new_task_graph.edges if e[0] == 0]
            source_out_edges_ids = [new_task_graph.edges.index(e) for e in source_out_edges]
            for (source_out_edge, source_out_edge_id) in zip(source_out_edges, source_out_edges_ids):
                og_nbr_task = new_to_old_node_mapping[source_out_edge[1]]
                last_graph_nbr_task = last_node_mapping.index(og_nbr_task)
                last_graph_in_edges = [e for e in last_graph.edges if e[1] == last_graph_nbr_task]
                last_graph_in_edge_inds = [last_graph.edges.index(e) for e in last_graph_in_edges]
                last_graph_total_inflow = np.sum([last_solution[i] for i in last_graph_in_edge_inds])
                last_graph_in_edges_og_names = [(last_node_mapping[e[0]], last_node_mapping[e[1]]) for e in last_graph_in_edges]
                for (edge, last_graph_edge_ind) in zip(last_graph_in_edges_og_names, last_graph_in_edge_inds):
                    if edge[0] in new_to_old_node_mapping:
                        edge_new_name = (new_to_old_node_mapping.index(edge[0]), new_to_old_node_mapping.index(edge[1]))
                        if edge_new_name in new_task_graph.edges and edge_new_name[0] != 0:
                            edge_new_id = new_task_graph.edges.index(edge_new_name)
                            proposed_solution[edge_new_id] = last_solution[last_graph_edge_ind]
                            last_graph_total_inflow -= last_solution[last_graph_edge_ind]
                            updated_edges[edge_new_id] = True
                            print(f"updated edge {edge_new_name} to {last_solution[last_graph_edge_ind]} ")

                proposed_solution[source_out_edge_id] = last_graph_total_inflow
                print(f"Added remaining excess inflow of {last_graph_total_inflow} to edge {source_out_edge}")
                updated_edges[source_out_edge_id] = True

            # TODO there is an error where an edge may not be updated when it goes to a node that received flow from a deleted node that was
            # not the source node. e.g. flow went 0 --> 1 --> 6, then nodes 0 and 1 are deleted. how to address??
            # if in the solution immediately prior to current solution, there is no flow incoming to node 6 from a deleted node,
            # then that node will not be connected to source. Therefore, we have no way of inputting the flow to node 6.
            # TO ADDRESS: are there any risks associated with connecting ALL nodes with deleted incoming edges to source?
            # I don't believe this will result in any precedence issues due to the way pruning works.
            # once that edge to source is added, can we iteratively just go through all nodes connected to source and ensure
            # that they have the correct amount of incoming flow??
            # map old graph edges onto new graph
            # iterate through edges in current step's graph, finding if there are matches in the old graph
            non_updated_edges = [e for e in new_task_graph.edges if not updated_edges[new_task_graph.edges.index(e)]]
            non_updated_edges_ids = [i for i in range(new_task_graph.num_edges) if not updated_edges[i]]

            for (edge, edge_id) in zip(non_updated_edges, non_updated_edges_ids):
                og_edge_name = (new_to_old_node_mapping[edge[0]], new_to_old_node_mapping[edge[1]])
                if -1 in og_edge_name:
                    #import pdb; pdb.set_trace()
                    raise Exception
                edge_name_in_last_graph = (last_node_mapping.index(og_edge_name[0]), last_node_mapping.index(og_edge_name[1]))
                if edge_name_in_last_graph in last_graph.edges:
                    last_graph_edge_id = last_graph.edges.index(edge_name_in_last_graph)
                    proposed_solution[edge_id] = last_solution[last_graph_edge_id]
                    updated_edges[edge_id] = True
            try:
                print(proposed_solution)
                proposed_assignment, _, proposed_fin_times = self.get_assignment(new_task_graph, proposed_solution, new_to_old_node_mapping, inprogress_coalitions_source_it=graph_ct)
                print(f"PROPOSED FIN TIMES: {np.array(proposed_fin_times) + self.current_time}")
                proposed_solution_reward_nodewise = new_task_graph.reward_model._nodewise_optim_cost_function(proposed_solution)
                for task in range(new_task_graph.num_tasks):
                    if proposed_fin_times[task] + self.current_time > self.original_task_graph.makespan_constraint:
                        print(f"Task {task} completed after makespan deadline: removing {-proposed_solution_reward_nodewise[task]} reward")
                        proposed_solution_reward_nodewise[task] = 0

                # if an inprogress task's assigned coalition size is changed, it yields zero reward
                for (inprogress_task, inprogress_id) in zip(new_task_graph.source_node_info_dict['in_progress'], range(len(new_task_graph.source_node_info_dict['in_progress']))):
                    original_task_coalition = new_task_graph.source_node_info_dict['node_capacities'][inprogress_id+1]
                    new_task_coalition = len(proposed_assignment[inprogress_task])/self.original_num_robots
                    if original_task_coalition != new_task_coalition:
                        print(f"Coalition assigned to inprogress task {inprogress_task} changed -- task cancelled and {proposed_solution_reward_nodewise[inprogress_task]} reward removed")
                        proposed_solution_reward_nodewise[inprogress_task] = 0.0
            except Exception as e:
                print(type(e))
                print(e)
                print(f"INFEASIBLE MAPPING FROM SOLUTION {graph_ct} to new graph {self.current_step}. Logging negative reward")
                proposed_solution_reward_nodewise = np.ones(new_task_graph.num_tasks)*10000000
                #import pdb; pdb.set_trace()
                # TODO still need to work on catching bugs via this except


            proposed_solution_reward = -np.sum(proposed_solution_reward_nodewise)
            print(f"Updated flow solution from graph {graph_ct}: {proposed_solution}")
            print(f"The following edges were updated: {updated_edges}")
            print(f"The updates were: {np.array(proposed_solution)- np.array(new_flow_solution)}")
            print(f"The updated solution reward is {proposed_solution_reward}, a diff of {proposed_solution_reward - old_reward}")
            old_solution_new_graph_rewards[graph_ct] = proposed_solution_reward
            old_solution_new_graph_rewards[graph_ct] = proposed_solution_reward
            old_solution_new_graph_list[graph_ct] = proposed_solution

            graph_ct -= 1

        best_soln_id = np.argmax(old_solution_new_graph_rewards)
        if old_solution_new_graph_rewards[best_soln_id] > old_reward:
            print(f"New solution has been replaced with updated solution from step {best_soln_id}")
            #import pdb; pdb.set_trace()
            return old_solution_new_graph_list[best_soln_id], best_soln_id
        print(f"New solution was found to be best. No replacement necessary.")
        return new_flow_solution, self.current_step-1

    def create_oracle_reward_model(self, sigma=0.0):

        # copy the dictionary
        new_args = copy.deepcopy(self.original_args)
        if 'run_minlp' in new_args.keys():
            new_args['run_minlp'] = False
        # fix any ordering discrepancies
        new_args['edges'] = copy.deepcopy(
            self.original_task_graph.reward_model.edges)  # list of lists rather than list of tuples
        new_args['dependency_params'] = copy.deepcopy(self.original_task_graph.dependency_params)
        new_args['dependency_types'] = copy.deepcopy(self.original_task_graph.dependency_types)

        # perturb all params by drawing from N(param, sigma*param)
        for (param, param_ind) in zip(new_args['coalition_params'], range(len(new_args['coalition_params']))):
            new_param = []
            for entry in param:
                new_param.append(np.random.normal(entry, abs(entry*sigma)))
            new_args['coalition_params'][param_ind] = new_param
        for (param, param_ind) in zip(new_args['dependency_params'], range(len(new_args['dependency_params']))):
            new_param = []
            for entry in param:
                new_param.append(np.random.normal(entry, abs(entry*sigma)))
            new_args['dependency_params'][param_ind] = new_param

        oracle_task_graph = TaskGraph(**new_args)

        return oracle_task_graph.reward_model

    def get_assignment(self, taskgraph, flow, new_to_old_node_mapping, inprogress_coalitions_source_it=None):
        if inprogress_coalitions_source_it is None:
            inprogress_coalitions_source_it = self.current_step-1
        ordered_nodes = list(nx.topological_sort(taskgraph.task_graph))
        frontier_nodes = []
        task_start_times = np.zeros((taskgraph.num_tasks,))
        task_finish_times = np.zeros((taskgraph.num_tasks,))
        agent_assignments = [[] for _ in range(taskgraph.num_tasks)]
        temp_agent_assignments = [[] for _ in range(taskgraph.num_tasks)]
        nodelist = list(range(taskgraph.num_tasks))
        frontier_nodes.append(nodelist[0])
        incomplete_nodes = []

        inprogress_assigned_agents = []
        if taskgraph.source_node_info_dict:
            # first, allocate appropriate agents to inprogress nodes
            for current_node in range(1, taskgraph.source_node_info_dict['num_source_nodes']):
                # should only ever be one out neighbor
                current_source_out_nbrs = [e[1] for e in taskgraph.edges if e[0]==current_node]
                out_nbr_og_name = new_to_old_node_mapping[current_source_out_nbrs[0]]
                task_start_times[current_node] = 0
                agent_assignments[current_node] = copy.deepcopy(self.task_assignment_history[inprogress_coalitions_source_it][out_nbr_og_name])
                temp_agent_assignments[current_node] = copy.deepcopy(self.task_assignment_history[inprogress_coalitions_source_it][out_nbr_og_name])
                for agent in agent_assignments[current_node]:
                    inprogress_assigned_agents.append(agent)
                ordered_nodes.remove(current_node)
                print(f"assigning agents {agent_assignments[current_node]} to node {current_node}")

        task_start_times[0] = 0.0
        agent_assignments[0] = [a for a in list(range(self.original_num_robots)) if a not in inprogress_assigned_agents]
        temp_agent_assignments[0] = [a for a in list(range(self.original_num_robots)) if a not in inprogress_assigned_agents]
        ordered_nodes.remove(0)

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
                    print(f"calculating task start time for node {current_node}")
                    task_start_times[int(current_node)] = max(
                        [task_finish_times[int(incoming_edges[i][0])] + taskgraph.inter_task_travel_times[incoming_edge_inds[i]] for i in range(len(incoming_edges)) if
                         not incoming_edges[i][0] in np.array(incomplete_nodes)])
                    for ind in incoming_edge_inds:
                        num_agents = int(round(flow[ind] * taskgraph.num_robots))
                        edge = taskgraph.reward_model.edges[ind]
                        #print(f"{num_agents} flowing from edge {edge}")
                        incoming_node = edge[0]
                        agent_list = []
                        print(f"pulling {num_agents} agents from node {incoming_node} to node {current_node}")
                        for _ in range(num_agents):
                            agent_list.append(temp_agent_assignments[incoming_node].pop(0))
                        temp_agent_assignments[current_node] += agent_list
                        agent_assignments[current_node] += agent_list

            else: # is a source node
                print("ERROR! WE SHOULDN'T BE ITERATING THROUGH ANY NODES THAT HAVE NO INCOMING EDGES!")
                raise(Exception)

            task_finish_times[int(current_node)] = task_start_times[int(current_node)] + taskgraph.task_times[
                int(current_node)]

        for node in incomplete_nodes:
            task_finish_times[int(node)] = 0.0
        return agent_assignments, task_start_times, task_finish_times

    def draw_original_taskgraph(self, flow_solution, graph_img_dir='real_time_debug'):

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
        fig, ax = plt.subplots()
        graph_img_file = graph_img_dir + '/old_graph.jpg'
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

    def draw_new_taskgraph(self, new_task_graph, new_to_old_node_mapping, new_flow_solution, graph_img_dir='real_time_debug'):
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
        graph_img_file = graph_img_dir + f'/new_graph_{self.current_step}.jpg'
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

    def print_trial_history(self):
        for i in range(self.current_step):
            agent_coalitions = [self.task_assignment_history[i][j] for j in self.task_done_history[i]]
            print(f"STEP {i}: task {self.task_done_history[i]} completed by agents {agent_coalitions}")
            for task in self.task_done_history[i]:
                print(f"task {task} with duration {self.original_task_graph.task_times[task]} completed at time {self.step_times[i]}")

    def check_itt(self):
        agent_prior_tasks = [0 for _ in range(self.original_num_robots)]
        agent_current_tasks = [0 for _ in range(self.original_num_robots)]
        for i in range(self.current_step):

            # update current tasks of agents who just finished a task
            for task in self.task_done_history[i]:
                for agent in range(self.original_num_robots):
                    if agent in self.task_assignment_history[i][task]:
                        agent_current_tasks[agent] = task

            # check all itt
            for agent in range(self.original_num_robots):
                last_fin = self.current_finish_times[agent_prior_tasks[agent]]
                cur_start = self.current_start_times[agent_current_tasks[agent]]
                #if cur_start - last_fin > self.original_task_graph.inter_task_travel_times[self.original_task_graph.edges.index(())]
