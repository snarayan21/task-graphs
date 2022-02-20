from taskgraph import TaskGraph
from file_utils import clean_dir_name
import toml
import argparse
import pathlib
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Do a whole experiment.')
    parser.add_argument('-cfg', default=None, help='Specify path to the experiment toml file')
    #parser.add_argument('-baseline', '-b', action='store_true', default=False, help='include -baseline flag to additionally solve graph with baseline optimizer')
    args = parser.parse_args()

    all_args = toml.load(args.cfg)
    exp_args = all_args['exp']

    #create overall experiment data directory if it does not exist
    experiment_data_dir = pathlib.Path("experiment_data/")
    experiment_data_dir.mkdir(parents=True, exist_ok=True)

    #get cleaned name for this experiment
    experiment_dir_name = exp_args['exp_name']
    experiment_dir_path, experiment_dir_name = clean_dir_name(experiment_dir_name, experiment_data_dir)

    #create cleaned experiment directory
    experiment_dir = experiment_data_dir / experiment_dir_name
    experiment_dir.mkdir(parents=True, exist_ok=False)

    #GENERATE RANDOM DAG with parameters num_nodes, max_width
    num_nodes = exp_args['num_nodes']
    max_width = exp_args['max_width']
    #TODO use max_width

    nx_task_graph = nx.DiGraph()
    node_list = [0]
    edge_list = [] #list of edges in form (x,y)
    old_node_frontier = [0]
    new_node_frontier = []
    converge_flag = False
    branching_likelihood = 0.4
    cur_nodes = 1 #current number of nodes in the graph
    while cur_nodes < num_nodes - 1:
        converge_flag = False #converge flag cannot carry over to new frontier
        for node in old_node_frontier:
            branch_rand = np.random.uniform(low=0.0, high=1.0)
            converge_rand = np.random.uniform(low=0.0, high=1.0)
            branch = branch_rand < branching_likelihood
            if branch: # add two nodes to graph and frontier, add two edges from current node to new nodes, remove old node from frontier
                node_list.append(cur_nodes)
                new_node_frontier.append(cur_nodes)
                edge_list.append((node, cur_nodes))
                cur_nodes+=1
                
                if converge_flag:
                    edge_list.append((node,new_node_frontier[-2]))
                else:
                    node_list.append(cur_nodes)
                    new_node_frontier.append(cur_nodes)
                    edge_list.append((node, cur_nodes))
                    cur_nodes+=1
                
            else:
                if converge_flag:
                    edge_list.append((node,new_node_frontier[-1]))
                else:
                    node_list.append(cur_nodes)
                    new_node_frontier.append(cur_nodes)
                    edge_list.append((node, cur_nodes))
                    cur_nodes+=1

            converge_flag = converge_rand < branching_likelihood and node != old_node_frontier[-1]
        old_node_frontier = new_node_frontier
        new_node_frontier = []
    #terminate all frontier nodes into the sink node
    node_list.append(cur_nodes)
    for node in old_node_frontier:
        edge_list.append((node,cur_nodes))

    nx_task_graph.add_nodes_from(node_list)
    nx_task_graph.add_edges_from(edge_list)
    print("Graph is DAG: ", nx.is_directed_acyclic_graph(nx_task_graph))
    print("Graph is connected: ", nx.has_path(nx_task_graph, 0, node_list[-1]))
    nx.draw(nx_task_graph)
    plt.show()


    import pdb; pdb.set_trace()
    #task_graph = TaskGraph(**track_args['exp'])


if __name__ == '__main__':
    main()