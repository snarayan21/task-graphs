import numpy as np
from pydrake.solvers.mathematicalprogram import MathematicalProgram
from pydrake.solvers.mathematicalprogram import Solve
from pydrake.symbolic import Expression
from pydrake.solvers.snopt import SnoptSolver, SnoptSolverDetails
import pydrake.math as math
from pydrake.math import exp
import matplotlib.pyplot as plt
import networkx as nx

def formulate_graph():
    """

    :return:
    """
    G = nx.DiGraph()
    G.add_nodes_from([1, 2, 3, 4])
    G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4)])
    return G

task_graph = formulate_graph()


def identity(f):
    """

    :return:
    """
    return f

def dimishing_return1(flow):
    NotImplementedError
def diminshing_return2(flow):
    NotImplementedError

def step_function1(flow):
    a = 10
    b = 6
    c = 1.5
    return a / (1 + exp(-1 * b * (flow - c)))

def step_function2(flow):
    a = 10
    b = 6
    c = 1.5
    return a / (1 + exp(-1 * b * (flow - c)))

def apply_reward(node_idx, flow):
    """

    :param node_idx:
    :param flow:
    :return:
    """

    if node_idx == 1:
        return 0.0 * flow
    elif node_idx == 2:
        return step_function1(flow)
    elif node_idx == 3:
        return step_function2(flow)
    elif node_idx == 4:
        return 1.0 * flow
    else:
        AssertionError("Something's wrong with node indices")

def flow_cost(f):
    """

    :param f:
    :return:
    """
    N = task_graph.number_of_nodes()

    D = nx.linalg.graphmatrix.incidence_matrix(task_graph, oriented=True).A
    D_incoming = np.maximum(D, 0)
    D_outgoing = np.maximum(-D, 0)

    # total incoming flow into each node
    incoming_flow = D_incoming @ f
    # compute reward for node
    node_rewards = np.zeros(N,dtype=object)
    for node_i in range(N):
        node_rewards[node_i] = apply_reward(node_i+1, incoming_flow[node_i])
    # import pdb
    # pdb.set_trace()
    # now copy over rewards to the edges and sum it up as the total neg cost
    return -np.sum(D_outgoing.T @ node_rewards)

    # cost_vector = np.array([2, 1, 1, 1])
    # return np.dot(cost_vector, f)
    #return cost

def main():
    """

    :return:
    """
    N = 4 # number of robots
    M = task_graph.number_of_nodes()
    E = task_graph.number_of_edges()

    # Opt variable
    prog = MathematicalProgram()
    f = prog.NewContinuousVariables(E)
    # Define the Opt program
    # Constraint1: Flow must be positive on all edges
    prog.AddConstraint(identity, lb=np.zeros(E), ub=np.inf*np.ones(E), vars=f)

    # Constraint2: The inflow must be equal to outflow at all edges
    # compute incidence matrix
    D = nx.linalg.graphmatrix.incidence_matrix(task_graph, oriented=True).A
    b = np.zeros(M)
    b[0] = -N
    b[-1] = N
    prog.AddLinearEqualityConstraint(D, b, f)
    # prog.AddConstraint(f[0] + f[2] == f[1] + f[3])
    # now for the cost
    prog.AddCost(flow_cost, vars=f)

    result = Solve(prog)
    print("Success? ", result.is_success())
    print('f* = ', result.GetSolution(f))
    print('optimal cost = ', result.get_optimal_cost())
    print('solver is: ', result.get_solver_id().name())
    aa = result.get_solver_details().info
    print('Solver Status: ', aa)


    # f_list = list(result.GetSolution(f))
    # import pdb
    # pdb.set_trace()
    # flow_dict = {task_graph.edges[i]: f_list[i] for i in range(len(task_graph.edges))}
    # print(flow_dict)
    # pos = nx.planar_layout(task_graph)
    # nx.draw_networkx_edge_labels(task_graph, pos, edge_labels=flow_dict)
    #
    # plt.axis('off')
    # plt.show()

def edge_rewards(node, task_graph):
    """

    :param task_graph:
    :return:
    """


if __name__ == '__main__':
    main()
