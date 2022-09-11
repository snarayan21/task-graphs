from taskgraph import TaskGraph
from file_utils import clean_dir_name
import toml
import argparse
import pathlib
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
import pathlib

class ExperimentGenerator():

    def __init__(self, cmd_args):
        all_args = toml.load(cmd_args.cfg)
        exp_args = all_args['exp']

        self.num_trials = exp_args['num_trials']
        self.run_ddp = exp_args['run_ddp']
        if 'run_minlp' in exp_args.keys():
            self.run_minlp = exp_args['run_minlp']
        else:
            self.run_minlp = True
        if 'num_robots' in exp_args.keys():
            self.num_robots = exp_args['num_robots']
        else:
            self.num_robots = 4
        if 'makespan_constraint' in exp_args.keys():
            self.makespan_constraint = exp_args['makespan_constraint']
        else:
            self.makespan_constraint = 1.0
        self.coalition_influence_aggregator = exp_args['coalition_influence_aggregator']
        self.nodewise_coalition_influence_agg_list = None # nodewise list of individual coalition influence aggregation functions ('sum' or 'product')

        # create overall experiment data directory if it does not exist
        self.experiment_data_dir = pathlib.Path("experiment_data/")
        self.experiment_data_dir.mkdir(parents=True, exist_ok=True)

        # store whether we are running experiment from specific exp file or generating a new one
        self.from_file = exp_args['from_file']
        if cmd_args.inputargs is not None:
            self.filename = cmd_args.inputargs
        else:
            self.filename = exp_args['filename']

        # get cleaned name for this experiment
        if cmd_args.outputpath is not None:
            experiment_dir_name = cmd_args.outputpath
        else:
            if 'exp_name' in exp_args.keys():
                experiment_dir_name = exp_args['exp_name']
            elif self.from_file:
                exp_path = pathlib.Path(self.filename)
                experiment_dir_name = exp_path.name.replace(".toml", "_exp")
                breakpoint()
            else:
                raise NotImplementedError("must provide experiment name or load experiment from file")
        experiment_dir_path, experiment_dir_name = clean_dir_name(experiment_dir_name, self.experiment_data_dir)

        # create cleaned experiment directory
        self.experiment_dir = self.experiment_data_dir / experiment_dir_name
        self.experiment_dir.mkdir(parents=True, exist_ok=False)

        # GENERATE RANDOM DAG with parameters num_layers, num_layer_nodes_max
        self.num_layers = exp_args['num_layers']
        self.num_layer_nodes_max = exp_args['num_layer_nodes_max']
        if 'num_nodes' in exp_args.keys():
            self.num_nodes = exp_args['num_nodes']
        else:
            self.num_nodes = None


    def run_trials(self):
        """Runs all trials for the given experiment. Creates their directories, runs the DDP and baseline solution,
         and saves the results to a file in each directory. Also stores the results in the ExperimentGenerator for
         convenience"""

        if self.from_file:
            filename_path = pathlib.Path(self.filename)
            filename_list = []
            if filename_path.is_file():
                self.num_trials = 1
                filename_list.append(self.filename)
            else:
                pathlist = filename_path.glob("*.toml")
                self.num_trials = 0
                for path in pathlist:
                    print(path)
                    filename_list.append(path.absolute())
                    self.num_trials += 1

        trial_arg_list = []
        results_dict_list = []


        for trial_ind in range(self.num_trials):
            if not self.from_file:
                # generate args for a trial within the parameters loaded into the experiment
                trial_args, nx_task_graph, node_pos = self.generate_taskgraph_args()
            else:
                trial_args, nx_task_graph, node_pos = load_taskgraph_args(filename_list[trial_ind])
            #create directory for results
            dir_name = "trial_" + str(trial_ind)
            trial_dir = self.experiment_dir / dir_name
            trial_dir.mkdir(parents=True, exist_ok=False)

            args_file = trial_dir / "args.toml"
            with open(args_file, "w") as f:
                toml.dump(trial_args,f)

            print("TRIAL GENERATED")
            task_graph = TaskGraph(**trial_args['exp'])
            print("TASK GRAPH INITIALIZED")

            graph_img_file = trial_dir / "graph.jpg"
            label_dict = {}
            for i in range(task_graph.num_tasks):
                label_dict[i] = str(i)
            #nx.draw_networkx_labels(nx_task_graph, labels=label_dict)
            nx.draw(nx_task_graph, labels=label_dict, pos=node_pos)
            plt.savefig(graph_img_file.absolute())
            plt.clf()
            start = time.time()

            #solve greedy
            task_graph.solve_graph_greedy()
            greedy_fin_time = time.time()
            greedy_elapsed_time = greedy_fin_time-start
            print("GREEDY SOLUTION FINISHED")
            run_ddp = self.run_ddp
            if run_ddp:
                #solve with ddp
                task_graph.initialize_solver_ddp(**trial_args['ddp'])
                task_graph.solve_ddp()
                ddp_fin_time = time.time()
                ddp_elapsed_time = ddp_fin_time-greedy_fin_time
                print("DDP SOLUTION FINISHED")

            #solve baseline
            task_graph.solve_graph_scipy()
            baseline_fin_time = time.time()
            print("BASELINE SOLUTION FINISHED")

            if run_ddp:
                baseline_elapsed_time = baseline_fin_time-ddp_fin_time
            else:
                baseline_elapsed_time = baseline_fin_time-greedy_fin_time #ddp_fin_time

            #solve minlp
            if self.run_minlp:
                task_graph.solve_graph_minlp()
            else:
                task_graph.solve_graph_minlp_dummy()
            minlp_fin_time = time.time()
            minlp_elapsed_time = minlp_fin_time - baseline_fin_time
            print("MINLP SOLUTION FINISHED")

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
            results_dict['baseline_task_rewards'] = -np.array(task_graph.reward_model._nodewise_optim_cost_function(task_graph.last_baseline_solution.x), dtype='float')
            results_dict['baseline_solution'] = task_graph.last_baseline_solution
            results_dict['baseline_solution_time'] = baseline_elapsed_time
            results_dict['baseline_makespan'] = np.max(task_graph.time_task_execution(task_graph.last_baseline_solution.x)[1])
            results_dict['baseline_execution_times'] = task_graph.time_task_execution(task_graph.last_baseline_solution.x)

            results_dict['pruned_baseline_solution'] = task_graph.pruned_baseline_solution.x
            results_dict['pruned_baseline_reward'] = -task_graph.reward_model.flow_cost(task_graph.pruned_baseline_solution.x)
            results_dict['pruned_baseline_makespan'] = np.max(task_graph.time_task_execution(task_graph.pruned_baseline_solution.x)[1])

            results_dict['pruned_rounded_baseline_solution'] = task_graph.pruned_rounded_baseline_solution
            results_dict['pruned_rounded_baseline_reward'] = -task_graph.reward_model.flow_cost(task_graph.pruned_rounded_baseline_solution)

            results_dict['rounded_baseline_solution'] = task_graph.rounded_baseline_solution
            results_dict['rounded_baseline_reward'] = -task_graph.reward_model.flow_cost(task_graph.rounded_baseline_solution)

            results_dict['greedy_reward'] = -task_graph.reward_model.flow_cost(task_graph.last_greedy_solution)
            results_dict['greedy_solution'] = task_graph.last_greedy_solution
            results_dict['greedy_solution_time'] = greedy_elapsed_time
            results_dict['greedy_makespan'] = np.max(task_graph.time_task_execution(task_graph.last_greedy_solution)[1])
            results_dict['greedy_execution_times'] = task_graph.time_task_execution(task_graph.last_greedy_solution)

            results_dict['pruned_greedy_solution'] = task_graph.pruned_greedy_solution
            results_dict['pruned_greedy_reward'] = -task_graph.reward_model.flow_cost(task_graph.pruned_greedy_solution)
            results_dict['pruned_greedy_makespan'] = np.max(task_graph.time_task_execution(task_graph.pruned_greedy_solution)[1])

            results_dict['pruned_rounded_greedy_solution'] = task_graph.pruned_rounded_greedy_solution
            results_dict['pruned_rounded_greedy_reward'] = -task_graph.reward_model.flow_cost(task_graph.pruned_rounded_greedy_solution)

            if np.isinf(np.array(results_dict['pruned_rounded_greedy_reward'],results_dict['pruned_rounded_baseline_reward'])).any():
                pass
                #breakpoint()
            if run_ddp:
                results_dict['ddp_reward'] = -task_graph.reward_model.flow_cost(task_graph.last_ddp_solution)
                results_dict['ddp_solution'] = task_graph.last_ddp_solution
                results_dict['ddp_solution_time'] = ddp_elapsed_time
                results_dict['ddp_makespan'] = task_graph.time_task_execution(task_graph.last_ddp_solution)[1][-1]
                results_dict['ddp_execution_times'] = task_graph.time_task_execution(task_graph.last_ddp_solution)



            results_dict['minlp_reward'] = task_graph.last_minlp_solution_val
            results_dict['minlp_solution'] = task_graph.last_minlp_solution
            results_dict['minlp_solution_time'] = minlp_elapsed_time
            results_dict['minlp_makespan'] = task_graph.minlp_makespan
            results_dict['minlp_execution_times'] = task_graph.last_minlp_solution[-trial_args['exp']['num_tasks']:]
            results_dict['minlp_primal_bound'] = task_graph.minlp_primal_bound
            results_dict['minlp_dual_bound'] = task_graph.minlp_dual_bound
            results_dict['minlp_gap'] = task_graph.minlp_gap
            results_dict['minlp_obj_limit'] = task_graph.minlp_obj_limit

            if self.run_minlp:
                results_dict['MINLP details'] = task_graph.translate_minlp_objective(task_graph.last_minlp_solution)


            results_dict_list.append(results_dict)

            results_file = trial_dir / "results.toml"
            with open(results_file, "w") as f2:
                toml.dump(results_dict,f2)

            # overwrite args file with task times
            trial_args['exp']['task_times'] = task_graph.task_times
            with open(args_file, "w") as f:
                toml.dump(trial_args,f)

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
        taskgraph_args_exp['edges'] = [np.array(edge) for edge in edge_list]
        taskgraph_args_exp['num_robots'] = self.num_robots
        taskgraph_args_exp['makespan_constraint'] = self.makespan_constraint
        taskgraph_args_exp['coalition_influence_aggregator'] = self.coalition_influence_aggregator #'product' # or 'sum'
        if self.coalition_influence_aggregator == 'sum':
            self.nodewise_coalition_influence_agg_list = ['sum' for _ in range(trial_num_nodes)]
        elif self.coalition_influence_aggregator == 'product':
            self.nodewise_coalition_influence_agg_list = ['product' for _ in range(trial_num_nodes)]
        elif self.coalition_influence_aggregator == 'mix':
            rand_inds = np.random.choice([0,1], size=trial_num_nodes)
            choices = ['sum','product']
            self.nodewise_coalition_influence_agg_list = [choices[rand_inds[i]] for i in range(trial_num_nodes)]
        else:
            raise(NotImplementedError())
        taskgraph_args_exp['nodewise_coalition_influence_agg_list'] = self.nodewise_coalition_influence_agg_list


        coalition_types_choices = ['sigmoid_b', 'dim_return', 'polynomial']
        coalition_types_indices = np.random.randint(0,3,(trial_num_nodes,))
        # sample from coalition types available iteratively to make list of strings
        taskgraph_args_exp['coalition_types'] = [coalition_types_choices[coalition_types_indices[i]] for i in range(trial_num_nodes)]

        # sample corresponding parameters within some defined range to the types in the above list
        #taskgraph_args_exp['coalition_params'] = [list(np.random.randint(-2,3,(3,))) for _ in range(trial_num_nodes)]
        M = 3
        coalition_params = []
        for i in range(trial_num_nodes):
            if taskgraph_args_exp['coalition_types'][i] == 'sigmoid_b':
                w = 0.25+np.random.random()*0.5
                k = np.random.randint(5,50)
                p0 = float( M*(1+np.exp(-k)*np.exp(k*w)) * (1+np.exp(k*w)) / ((1-np.exp(-k))*np.exp(k*w)) )
                p3 = float( M*(1+np.exp(-k)*np.exp(k*w)) / ((1-np.exp(-k))*np.exp(k*w)))
                coalition_params.append([p0,float(k),w,p3])
            if taskgraph_args_exp['coalition_types'][i] == 'dim_return':
                w = np.random.random()*10+0.5
                p0 = float(M/(1-np.exp(w)))
                p1 = w
                coalition_params.append([p0,p1,p0])
            if taskgraph_args_exp['coalition_types'][i] == 'polynomial':
                w = 0.5+np.random.random()/2 # vertex value in [0.5,1]
                p1 =  2*M/w
                p2 = -M/(w**2)
                coalition_params.append([0.000,p1,p2])

        taskgraph_args_exp['coalition_params'] = coalition_params

        # sample from dependency types available -- list of strings
        influence_types_choices = ['sigmoid_b', 'dim_return']
        influence_types_indices = np.random.randint(0,2,(num_edges,)) # TODO IMPLEMENT SIGMOID
        taskgraph_args_exp['dependency_types'] = [influence_types_choices[influence_types_indices[i]] for i in range(num_edges)]

        # sample corresponding parameters within some defined range to the types in the above list
        dependency_params = []
        for i in range(num_edges):
            if taskgraph_args_exp['dependency_types'][i] == 'sigmoid_b':
                w = 0.25+np.random.random()*5
                k = np.random.randint(1,10)
                p0 = float(M*(1+np.exp(k*w))/np.exp(k*w))
                p3 = float(M/np.exp(k*w))
                dependency_params.append([p0,float(k),w,p3])
            if taskgraph_args_exp['dependency_types'][i] == 'dim_return':
                w = np.random.random()*10+0.5
                dependency_params.append([float(M),w,float(M)])
        taskgraph_args_exp['dependency_params'] = dependency_params
        # sample from available agg types -- probably all sum for now???
        taskgraph_args_exp['aggs'] = ['or' for _ in range(trial_num_nodes)]

        taskgraph_args_exp['minlp_time_constraint'] = True #TODO make this a parameter
        taskgraph_args_exp['run_minlp'] = self.run_minlp
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
        num_layers = self.num_layers
        num_layer_nodes_max = self.num_layer_nodes_max #must be > 1
        require_precise_nodes = self.num_nodes is not None
        precise_nodes_condition = False
        while not precise_nodes_condition:
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

            if require_precise_nodes:
                precise_nodes_condition = trial_num_nodes == self.num_nodes
            else:
                precise_nodes_condition = True

        return nx_task_graph, edge_list, trial_num_nodes, frontiers_list


def load_taskgraph_args(filename):

    args = toml.load(filename)

    num_tasks = int(args['exp']['num_tasks'])
    edges = args['exp']['edges']

    task_graph = nx.DiGraph()
    task_graph.add_nodes_from(range(num_tasks))
    task_graph.add_edges_from(edges)

    return args, task_graph, None


def main():

    parser = argparse.ArgumentParser(description='Do a whole experiment.')
    parser.add_argument('-cfg', default=None, help='Specify path to the experiment toml file')
    parser.add_argument('-outputpath', '-o', default=None, help='Base name of output experiment dir')
    parser.add_argument('-inputargs', default=None, help='Specify filepath of an existing args.toml file to use input task graph instead of generating a random task graph')
    #parser.add_argument('-baseline', '-b', action='store_true', default=False, help='include -baseline flag to additionally solve graph with baseline optimizer')
    args = parser.parse_args()

    experiment_generator = ExperimentGenerator(args)
    trial_args, results = experiment_generator.run_trials()

    exp_num_tasks = [trial_args[i]['exp']['num_tasks'] for i in range(len(trial_args))]
    total_rewards = [[],[],[],[]] # task graph, greedy, MINLP, rounded_task_graph
    makespan = [[],[],[]]
    sol_time = [[],[],[]]
    for i in range(len(results)):
        result_dict = results[i]

        total_rewards[0].append(result_dict['pruned_rounded_baseline_reward'])
        makespan[0].append(result_dict['baseline_makespan'])
        sol_time[0].append(result_dict['baseline_solution_time'])
        total_rewards[1].append(result_dict['pruned_rounded_greedy_reward'])
        makespan[1].append(result_dict['greedy_makespan'])
        sol_time[1].append(result_dict['greedy_solution_time'])
        total_rewards[2].append(result_dict['minlp_reward'])
        makespan[2].append(result_dict['minlp_makespan'])
        sol_time[2].append(result_dict['minlp_solution_time'])
        total_rewards[3].append(result_dict['rounded_baseline_reward'])

    # PLOT
    fig, axs = plt.subplots(4,1,sharex=True, figsize=(6,9))
    labels = ["Task Graph", "Greedy", "MINLP", "Task Graph Discrete"]
    for k in range(4):
        axs[0].plot(total_rewards[k], label=labels[k])
        axs[0].set_ylabel('Reward')
        axs[0].set_ylim([-5, None])
        axs[0].legend()

        rel_rewards = [total_rewards[k][i]/total_rewards[0][i] for i in range(len(total_rewards[0]))]
        axs[3].plot(rel_rewards,label=labels[k])
        axs[3].set_ylabel('Relative Reward')
        axs[3].set_ylim([0.0, None])
        axs[3].legend()

    for k in range(3):

        axs[1].plot(makespan[k], label=labels[k])
        axs[1].set_ylabel('Makespan')
        axs[1].legend()

        axs[2].plot(sol_time[k], label=labels[k])
        axs[2].set_ylabel('Solution Time')
        axs[2].legend()

    plt.savefig((experiment_generator.experiment_dir / 'exp_results.png').absolute())

    plt.clf()   # clear plot for next iteration

if __name__ == '__main__':
    main()
