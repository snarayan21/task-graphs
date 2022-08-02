from taskgraph import TaskGraph
from file_utils import clean_dir_name
import toml
import argparse
import pathlib
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time

class ExperimentGenerator():

    def __init__(self,cmd_args):
        all_args = toml.load(cmd_args.cfg)
        exp_args = all_args['exp']

        self.num_trials = exp_args['num_trials']

        # create overall experiment data directory if it does not exist
        self.experiment_data_dir = pathlib.Path("experiment_data/")
        self.experiment_data_dir.mkdir(parents=True, exist_ok=True)

        # get cleaned name for this experiment
        experiment_dir_name = exp_args['exp_name']
        experiment_dir_path, experiment_dir_name = clean_dir_name(experiment_dir_name, self.experiment_data_dir)

        # create cleaned experiment directory
        self.experiment_dir = self.experiment_data_dir / experiment_dir_name
        self.experiment_dir.mkdir(parents=True, exist_ok=False)

        # GENERATE RANDOM DAG with parameters num_nodes, max_width
        self.num_nodes = exp_args['num_nodes'] #NOTE num_nodes may not be the EXACT number of nodes in the graph - varies by 1 or 2
        self.max_width = exp_args['max_width']
        # TODO use max_width

    def run_trials(self):
        """Runs all trials for the given experiment. Creates their directories, runs the DDP and baseline solution,
         and saves the results to a file in each directory. Also stores the results in the ExperimentGenerator for
         convenience"""

        trial_arg_list = []
        results_dict_list = []


        for trial_ind in range(self.num_trials):

            # generate args for a trial within the parameters loaded into the experiment
            trial_args, nx_task_graph, node_pos = self.generate_taskgraph_args()

             #create directory for results
            dir_name = "trial_" + str(trial_ind)
            trial_dir = self.experiment_dir / dir_name
            trial_dir.mkdir(parents=True, exist_ok=False)

            args_file = trial_dir / "args.toml"
            with open(args_file, "w") as f:
                toml.dump(trial_args,f)

            task_graph = TaskGraph(**trial_args['exp'])



            graph_img_file = trial_dir / "graph.jpg"
            label_dict = {}
            for i in range(task_graph.num_tasks):
                label_dict[i] = str(i)
            #nx.draw_networkx_labels(nx_task_graph, labels=label_dict)
            nx.draw(nx_task_graph, labels=label_dict, pos=node_pos)
            plt.savefig(graph_img_file.absolute())
            breakpoint()
            start = time.time()

            #solve greedy
            task_graph.solve_graph_greedy()
            greedy_fin_time = time.time()
            greedy_elapsed_time = greedy_fin_time-start

            run_ddp = False
            if run_ddp:
                #solve with ddp
                task_graph.initialize_solver_ddp(**trial_args['ddp'])
                task_graph.solve_ddp()
                ddp_fin_time = time.time()
                ddp_elapsed_time = ddp_fin_time-greedy_fin_time

            #solve baseline
            task_graph.solve_graph_scipy()
            baseline_fin_time = time.time()
            if run_ddp:
                baseline_elapsed_time = baseline_fin_time-ddp_fin_time
            else:
                baseline_elapsed_time = baseline_fin_time-greedy_fin_time #ddp_fin_time

            #solve minlp
            task_graph.solve_graph_minlp()
            minlp_fin_time = time.time()
            minlp_elapsed_time = minlp_fin_time - baseline_fin_time

            if run_ddp:
                ddp_data = trial_dir / "ddp_data.jpg"
                fig, axs = plt.subplots(4,1,sharex=True, figsize=(6,12))
                axs[0].plot(task_graph.ddp_reward_history)
                axs[0].set_ylabel('Reward')

                axs[1].plot(task_graph.constraint_residual)
                axs[1].set_ylabel('Constraint Residual Norm')

                axs[2].plot(task_graph.alpha_hist)
                axs[2].set_ylabel('Alpha value')

                axs[3].plot(task_graph.buffer_hist)
                axs[3].set_ylabel('Buffer value')

                fig.text(0.5, 0.04, 'Iteration #', ha='center')
                plt.savefig(ddp_data.absolute())

                plt.clf()   # clear plot for next iteration

            #log results
            trial_arg_list.append(trial_args)
            results_dict = {}
            results_dict['baseline_reward'] = -task_graph.reward_model.flow_cost(task_graph.last_baseline_solution.x)
            results_dict['baseline_solution'] = task_graph.last_baseline_solution
            results_dict['baseline_solution_time'] = baseline_elapsed_time
            results_dict['baseline_execution_times'] = task_graph.time_task_execution(task_graph.last_baseline_solution.x)

            results_dict['greedy_reward'] = -task_graph.reward_model.flow_cost(task_graph.last_greedy_solution)
            results_dict['greedy_solution'] = task_graph.last_greedy_solution
            results_dict['greedy_solution_time'] = greedy_elapsed_time
            results_dict['greedy_execution_times'] = task_graph.time_task_execution(task_graph.last_greedy_solution)

            if run_ddp:
                results_dict['ddp_reward'] = -task_graph.reward_model.flow_cost(task_graph.last_ddp_solution)
                results_dict['ddp_solution'] = task_graph.last_ddp_solution
                results_dict['ddp_solution_time'] = ddp_elapsed_time
                results_dict['ddp_execution_times'] = task_graph.time_task_execution(task_graph.last_ddp_solution)


            results_dict['minlp_reward'] = task_graph.last_minlp_solution_val
            results_dict['minlp_solution'] = task_graph.last_minlp_solution
            results_dict['minlp_solution_time'] = minlp_elapsed_time
            results_dict['minlp_execution_times'] = task_graph.last_minlp_solution[-trial_args['exp']['num_tasks']:]


            results_dict_list.append(results_dict)

            results_file = trial_dir / "results.toml"
            with open(results_file, "w") as f2:
                toml.dump(results_dict,f2)

        return trial_arg_list, results_dict_list




    def generate_taskgraph_args(self):
        nx_task_graph, edge_list, trial_num_nodes, frontiers_list = self.generate_graph_topology_b()
        num_edges = len(edge_list)

        #nx.draw(nx_task_graph)
        #plt.show()

        #generate node positions
        node_pos = {}
        x_tot = 4
        y_tot = 3
        n_frontiers = len(frontiers_list)+1
        node_pos[0] = (0,y_tot/2)
        frontier_ct = 1
        for frontier in frontiers_list:
            x_pos = frontier_ct*(1/n_frontiers)*x_tot
            frontier_ct += 1
            n_nodes = len(frontier)
            y_interval = y_tot/n_nodes
            node_ct = 1
            for node in frontier:
                node_pos[node] = (x_pos,y_tot-node_ct*y_interval)
                node_ct += 1

        taskgraph_args = {}
        taskgraph_args_exp = {}
        taskgraph_args_exp['max_steps'] = 100
        taskgraph_args_exp['num_tasks'] = trial_num_nodes
        taskgraph_args_exp['edges'] = edge_list
        taskgraph_args_exp['numrobots'] = 1

        # sample from coalition types available iteratively to make list of strings
        taskgraph_args_exp['coalition_types'] = ['polynomial' for _ in range(trial_num_nodes)]

        # sample corresponding parameters within some defined range to the types in the above list
        #taskgraph_args_exp['coalition_params'] = [list(np.random.randint(-2,3,(3,))) for _ in range(trial_num_nodes)]
        coalition_params = []
        for _ in range(trial_num_nodes):
            c = 0.0
            b = np.random.uniform(0,3)
            a = np.random.uniform(-b,3)
            coalition_params.append([c,b,a])

        taskgraph_args_exp['coalition_params'] = coalition_params
        # sample from dependency types available -- list of strings
        taskgraph_args_exp['dependency_types'] = ['polynomial' for _ in range(num_edges)]

        # sample corresponding parameters within some defined range to the types in the above list
        #taskgraph_args_exp['dependency_params'] = [list(np.random.uniform(-.2,.2,(3,))) for _ in range(num_edges)]
        dependency_params = []
        for _ in range(num_edges):
            c = np.random.uniform(-0.2,0.2)
            b = np.random.uniform(0,3)
            a = np.random.uniform(-b,3)
            dependency_params.append([c,b,a])
        taskgraph_args_exp['dependency_params'] = dependency_params
        # sample from available agg types -- probably all sum for now???
        taskgraph_args_exp['aggs'] = ['or' for _ in range(trial_num_nodes)]

        taskgraph_args_exp['minlp_time_constraint'] = True #TODO make this a parameter

        taskgraph_args['exp'] = taskgraph_args_exp
        taskgraph_args['ddp'] = {'constraint_type': 'qp',
                                 'constraint_buffer': 'soft', #None or 'soft' or 'hard'
                                 'alpha_anneal': 'True', #'True' or 'False'
                                 'flow_lookahead': 'False' #'True' or 'False'
                                 }

        return taskgraph_args, nx_task_graph, node_pos

    def generate_graph_topology_a(self):
        nx_task_graph = nx.DiGraph()
        node_list = [0]
        edge_list = []  # list of edges in form (x,y)


        old_node_frontier = [0]
        new_node_frontier = []
        frontiers_list = []
        converge_flag = False
        branching_likelihood = 0.4
        cur_nodes = 1  # current number of nodes in the graph
        while cur_nodes < self.num_nodes - 1:
            converge_flag = False  # converge flag cannot carry over to new frontier
            for node in old_node_frontier:
                branch_rand = np.random.uniform(low=0.0, high=1.0)
                converge_rand = np.random.uniform(low=0.0, high=1.0)
                branch = branch_rand < branching_likelihood
                if branch:  # add two nodes to graph and frontier, add two edges from current node to new nodes, remove old node from frontier
                    node_list.append(cur_nodes)
                    new_node_frontier.append(cur_nodes)
                    edge_list.append((node, cur_nodes))
                    cur_nodes += 1

                    if converge_flag:
                        edge_list.append((node, new_node_frontier[-2]))
                    else:
                        node_list.append(cur_nodes)
                        new_node_frontier.append(cur_nodes)
                        edge_list.append((node, cur_nodes))
                        cur_nodes += 1

                else:
                    if converge_flag:
                        edge_list.append((node, new_node_frontier[-1]))
                    else:
                        node_list.append(cur_nodes)
                        new_node_frontier.append(cur_nodes)
                        edge_list.append((node, cur_nodes))
                        cur_nodes += 1

                converge_flag = converge_rand < branching_likelihood and node != old_node_frontier[-1]
            old_node_frontier = new_node_frontier
            frontiers_list.append(new_node_frontier)
            new_node_frontier = []
        # terminate all frontier nodes into the sink node

        node_list.append(cur_nodes)
        frontiers_list.append([cur_nodes])
        for node in old_node_frontier:
            edge_list.append((node, cur_nodes))

        trial_num_nodes = len(node_list)
        nx_task_graph.add_nodes_from(node_list)
        nx_task_graph.add_edges_from(edge_list)
        print("Graph is DAG: ", nx.is_directed_acyclic_graph(nx_task_graph))
        print("Graph is connected: ", nx.has_path(nx_task_graph, 0, node_list[-1]))

        return nx_task_graph, edge_list, trial_num_nodes, frontiers_list

    def generate_graph_topology_b(self):
        num_layers = 3
        num_layer_nodes_max = 2 #must be > 1

        nx_task_graph = nx.DiGraph()
        node_list = [0]
        edge_list = []  # list of edges in form (x,y)
        old_layer = [0]
        frontiers_list = []

        cur_layer = 0  # current number of nodes in the graph
        cur_node = 1
        #layer 0
        for node in range(np.random.randint(1, num_layer_nodes_max+1)):
            node_list.append(cur_node)
            old_layer.append(cur_node)
            edge_list.append((0, cur_node))
            cur_node += 1

        frontiers_list.append([0])
        frontiers_list.append(old_layer[1:])

        for layer in range(1,num_layers):
            #generate 1 to num_layer_nodes_max nodes for new layer
            new_layer_size = np.random.randint(1, num_layer_nodes_max+1)
            new_layer = np.arange(cur_node,cur_node + new_layer_size)
            for new_node in new_layer:
                node_list.append(new_node)

            #generate connection from old layer to new layer
            #   guarantees all old_layer nodes are connected to 1+ nodes in new layer
            connected_new_nodes = np.array([])
            for old_node in old_layer:
                #TODO :
                """connections can be changed to control the complexity of the graph.
                Options include limiting num_connections, sampling num_connections
                non-uniformly (e.g., more likely to have only one connection), or 
                choosing connection in a manner that considers all of old layer rather
                than on a node by node basis. (None of these methods are implemented yet)
                """
                num_edges = np.random.randint(1,new_layer_size+1) #from old_node
                edge_destinations = np.random.choice(new_layer, num_edges, replace=False)
                connected_new_nodes = np.append(connected_new_nodes, edge_destinations)
                for edge_destination in edge_destinations:
                    edge_list.append((int(old_node), int(edge_destination)))

            #check if new layer nodes all have connected, if not, connect to earlier nodes
            unconnected_new_nodes = np.setdiff1d(new_layer, np.unique(connected_new_nodes).astype(int))
            for node in unconnected_new_nodes:
                edge_origin = np.random.choice(np.arange(cur_node))
                edge_list.append((int(edge_origin), node))

            cur_node = cur_node + new_layer_size
            old_layer = new_layer
            frontiers_list.append(old_layer)

        # terminate all frontier nodes into the sink node
        sink = cur_node
        node_list.append(sink)
        frontiers_list.append([sink])
        for node in old_layer:
            edge_list.append((int(node), sink))

        num_edges = len(edge_list)
        trial_num_nodes = len(node_list)
        nx_task_graph.add_nodes_from(node_list)
        nx_task_graph.add_edges_from(edge_list)
        print("Graph is DAG: ", nx.is_directed_acyclic_graph(nx_task_graph))
        print("Graph is connected: ", nx.has_path(nx_task_graph, 0, node_list[-1]))
        breakpoint()
        return nx_task_graph, edge_list, trial_num_nodes, frontiers_list


def main():

    parser = argparse.ArgumentParser(description='Do a whole experiment.')
    parser.add_argument('-cfg', default=None, help='Specify path to the experiment toml file')
    #parser.add_argument('-baseline', '-b', action='store_true', default=False, help='include -baseline flag to additionally solve graph with baseline optimizer')
    args = parser.parse_args()

    experiment_generator = ExperimentGenerator(args)
    trial_args, results = experiment_generator.run_trials()

if __name__ == '__main__':
    main()
