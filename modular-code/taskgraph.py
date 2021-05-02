import numpy as np
from pydrake.solvers.mathematicalprogram import MathematicalProgram
from pydrake.solvers.mathematicalprogram import Solve
import pydrake.math as math
import matplotlib.pyplot as plt
import networkx as nx


class TaskGraph:
    # class for task graphs where nodes are tasks and edges are precedence relationships

    def __init__(self, num_tasks, edges, coalition_params, coalition_types, dependency_params, dependency_types, aggs,
                 numrobots):
        self.num_tasks = num_tasks
        self.num_robots = numrobots
        self.edges = edges
        self.task_graph = nx.DiGraph()
        self.task_graph.add_nodes_from(range(num_tasks))
        self.task_graph.add_edges_from(edges)
        self.num_edges = len(edges)  # number of edges

        self.coalition_params = coalition_params
        self.coalition_types = coalition_types
        self.dependency_params = dependency_params
        self.dependency_types = dependency_types
        self.aggs = aggs

    def identity(self, f):
        """
        :return:
        """
        return f

    def compute_node_coalition(self, node_i, f):
        """

        :param node_i:
        :param f:
        :return:
        """
        if node_i != 0 or node_i != self.num_tasks:
            coalition_function = getattr(self, self.coalition_types[node_i])
            return coalition_function(f, param=self.coalition_params[node_i])
        else:
            # source and sink node has 0 coalition/reward
            return 0

    def compute_node_reward(self, node_i):
        """
        #TODO: Encode task dependencies

        :param node_i:
        :param f:
        :return:
        """
        # compute incoming edges to node_i
        incoming_edges = list(self.task_graph.in_edges(node_i))

        task_influence_value = []
        for edge in incoming_edges:
            # find global index of edge
            edge_id = list(self.task_graph.edges).index(edge)
            # find the source node of that edge
            source_node = self.edges[edge_id][0]
            # extract the delta function applicable to this edge
            task_interdep = getattr(self, self.dependency_types[edge_id])
            # compute the task influence value (delta for an edge)
            task_influence_value.append(task_interdep(self.node_reward[source_node],
                                                      self.dependency_params[edge_id]))
        #TODO (Walker): Below is a specific choice of aggregation and combination with coalition function
        reward = self.node_coalition[node_i]

        for val in task_influence_value:
            reward = reward * val

        # reward = self.node_coalition[node_i]
        return reward

    def flow_cost(self, f):
        """

        :param f:
        :return:
        """
        D_incoming = np.maximum(self.incidence_mat, 0)

        # total incoming flow into each node
        incoming_flow = D_incoming @ f

        # compute rewards for nodes (sequentially)
        self.node_reward = np.zeros(self.num_tasks, dtype=object)
        self.node_coalition = np.zeros(self.num_tasks, dtype=object)
        for node_i in range(self.num_tasks):
            # Compute Coalition Function
            self.node_coalition[node_i] = self.compute_node_coalition(node_i, incoming_flow[node_i])
            # Compute the reward by combining with Inter-Task Dependency Function
            # influencing nodes of node i
            self.node_reward[node_i] = self.compute_node_reward(node_i)

        # now copy over rewards to the edges and sum it up as the total neg cost
        return -np.sum(self.node_reward)

    def initializeSolver(self):
        '''
        This function will define variables, functions, and bounds based on the input info
        :return:
        '''

        self.prog = MathematicalProgram()
        self.flow_var = self.prog.NewContinuousVariables(self.num_edges)

        # Define the Opt program
        # Constraint1: Flow must be positive on all edges
        self.prog.AddConstraint(self.identity,
                                lb=np.zeros(self.num_edges),
                                ub=np.inf * np.ones(self.num_edges),
                                vars=self.flow_var)

        # Constraint2: The inflow must be equal to outflow at all edges
        # compute incidence matrix
        self.incidence_mat = nx.linalg.graphmatrix.incidence_matrix(self.task_graph, oriented=True).A
        b = np.zeros(self.num_tasks)
        b[0] = -self.num_robots
        b[-1] = self.num_robots
        self.prog.AddLinearEqualityConstraint(self.incidence_mat, b, self.flow_var)

        # now for the cost
        self.prog.AddCost(self.flow_cost, vars=self.flow_var)

    def solveGraph(self):
        result = Solve(self.prog)
        print("Success? ", result.is_success())
        print('f* = ', result.GetSolution(self.flow_var))
        print('optimal cost = ', result.get_optimal_cost())
        print('solver is: ', result.get_solver_id().name())
        print('Reward at each node is:')
        for i in range(self.num_tasks):
            print(self.compute_node_reward(i))

    def sigmoid(self, flow, param):
        return param[0] / (1 + np.exp(-1 * param[1] * (flow - param[2])))

    def dim_return(self, flow, param):
        return param[0] + (param[2] * (1 - np.exp(-1 * param[1] * flow)))

    # def mult(vars):
    #     return np.prod(vars)
    #
    # def add(vars):
    #     return np.sum(vars)
    #
    # def combo(vars):
    #     return np.prod(vars) * np.sum(vars)
    #
    #     # edges_tot will include all edges and their reverses
    #     self.edges_tot = self.edges.copy()
    #
    #     for e in self.edges:
    #         self.edges_tot.append([e[1], e[0]])
    #
    #     self.edge_dict_n = {}
    #     self.edge_dict_r = {}
    #     self.edge_dict_ndex = {}
    #     self.edge_dict_rdex = {}
    #
    #     for i in range(len(self.edges)):
    #         e = self.edges[i]
    #         if e[0] in self.edge_dict_n:
    #             self.edge_dict_n[e[0]].append(e[1])
    #         else:
    #             self.edge_dict_n[e[0]] = [e[1]]
    #
    #         if e[0] in self.edge_dict_ndex:
    #             self.edge_dict_ndex[e[0]].append(i)
    #         else:
    #             self.edge_dict_ndex[e[0]] = [i]
    #
    #         if e[1] in self.edge_dict_r:
    #             self.edge_dict_r[e[1]].append(e[0])
    #         else:
    #             self.edge_dict_r[e[1]] = [e[0]]
    #
    #         if e[1] in self.edge_dict_rdex:
    #             self.edge_dict_rdex[e[1]].append(i)
    #         else:
    #             self.edge_dict_rdex[e[1]] = [i]
    #
    #     for i in range(self.numnodes):
    #         if i not in self.edge_dict_n:
    #             self.edge_dict_n[i] = []
    #         if i not in self.edge_dict_r:
    #             self.edge_dict_r[i] = []
    #         if i not in self.edge_dict_ndex:
    #             self.edge_dict_ndex[i] = []
    #         if i not in self.edge_dict_rdex:
    #             self.edge_dict_rdex[i] = []
    #
    #     print(self.edge_dict_ndex)
    #     print(self.edge_dict_rdex)
    #
    #     # f is the flow across each edge
    #     self.f = self.prog.NewContinuousVariables(len(self.edges_tot), "f")
    #     # p is the coalition component of the reward function for each node
    #     self.p = self.prog.NewContinuousVariables(self.numnodes, "p")
    #     # d is the previous reward component of the reward function for each edge
    #     self.d = self.prog.NewContinuousVariables(len(self.edges), "d")
    #     # r is the reward for each node
    #     self.r = self.prog.NewContinuousVariables(self.numnodes, "r")
    #     # c is the combined flow coming into each node
    #     self.c = self.prog.NewContinuousVariables(self.numnodes, "c")
    #     # g is the aggregation of the deltas coming into each node
    #     self.g = self.prog.NewContinuousVariables(self.numnodes, "g")
    #
    #     # all these variables must be positive
    #     for i in range(self.numnodes):
    #         self.prog.AddConstraint(self.g[i] >= 0)
    #         self.prog.AddConstraint(self.c[i] >= 0)
    #         self.prog.AddConstraint(self.r[i] >= 0)
    #         self.prog.AddConstraint(self.p[i] >= 0)
    #
    #     for i in range(len(self.edges)):
    #         self.prog.AddConstraint(self.d[i] >= 0)
    #
    #     for i in range(len(self.edges)):
    #         # flow cannot exceed number of robots
    #         self.prog.AddConstraint(self.f[i] <= self.numrobots)
    #         # flow over normal edges is inverse of flow on reverse edges
    #         self.prog.AddConstraint(self.f[i] == -1 * self.f[i + len(self.edges)])
    #
    #     for i in range(self.numnodes):
    #         inflow = []
    #         for j in self.edge_dict_rdex[i]:
    #             inflow.append(self.f[j])
    #
    #         inflow = np.array(inflow)
    #
    #         # c[i] is the inflow to node i -- important for rho function
    #         self.prog.AddConstraint(self.c[i] == np.sum(inflow))
    #
    #     # set the inflow of source node to 0
    #     self.prog.AddConstraint(self.c[0] == 0)
    #
    #     for i in range(1, self.numnodes - 1):
    #         outflow = []
    #         for j in self.edge_dict_ndex[i]:
    #             outflow.append(self.f[j])
    #
    #         outflow = np.array(outflow)
    #
    #         # c[i], which is node inflow, must be equal to node outflow (flow conservation)
    #         # this does not apply to the source or the sink
    #         self.prog.AddConstraint(self.c[i] - np.sum(outflow) == 0)
    #
    #     # outflow on node 0 (source) must be equal to number of robots
    #     source_outflow = []
    #     for i in self.edge_dict_ndex[0]:
    #         source_outflow.append(self.f[i])
    #     source_outflow = np.array(source_outflow)
    #     self.prog.AddConstraint(np.sum(source_outflow) == self.numrobots)
    #
    #     # inflow on last node (sink) must be equal to number of robots
    #     self.prog.AddConstraint(self.c[self.numnodes - 1] == self.numrobots)
    #
    #     # reward for source node is just a constant -- 1
    #     self.prog.AddConstraint(self.p[0] == 1)
    #     self.prog.AddConstraint(self.g[0] == 0)
    #     self.prog.AddConstraint(self.r[0] == 1)
    #
    #     # define rho functions as a function of node inflow
    #     for i in range(1, len(self.rhos) + 1):
    #         rho = self.rhos[i - 1]
    #         rhotype = self.rhotypes[i - 1]
    #         if (rhotype == "s"):
    #             print("node", i, ":", *rho)
    #             self.prog.AddConstraint(self.p[i] == TaskGraph.step(rho[0], rho[1], rho[2], self.c[i]))
    #         else:
    #             self.prog.AddConstraint(self.p[i] == TaskGraph.dimin(rho[0], rho[1], rho[2], self.c[i]))
    #
    #     # define delta functions as a function of previous reward
    #     for i in range(len(self.edges)):
    #         delta = self.deltas[i]
    #         deltatype = self.deltatypes[i]
    #         e = self.edges[i]
    #         if (deltatype == "s"):
    #             self.prog.AddConstraint(self.d[i] == TaskGraph.step(delta[0], delta[1], delta[2], self.r[e[0]]))
    #         else:
    #             self.prog.AddConstraint(self.d[i] == TaskGraph.dimin(delta[0], delta[1], delta[2], self.r[e[0]]))
    #
    #     # define agg functions as functions of incoming deltas
    #     for i in range(1, self.numnodes):
    #         agg = self.aggs[i - 1]
    #         indeltas = []
    #         for j in self.edge_dict_rdex[i]:
    #             indeltas.append(self.d[j])
    #
    #         indeltas = np.array(indeltas)
    #
    #         if (agg == "a"):
    #             self.prog.AddConstraint(self.g[i] == TaskGraph.add(indeltas))
    #
    #         elif (agg == "m"):
    #             self.prog.AddConstraint(self.g[i] == TaskGraph.mult(indeltas))
    #
    #         else:
    #             self.prog.AddConstraint(self.g[i] == TaskGraph.combo(indeltas))
    #
    #     # define reward as "combo" of rho and agg
    #     for i in range(1, self.numnodes):
    #         self.prog.AddConstraint(self.r[i] == self.g[i] * self.p[i] * (self.g[i] + self.p[i]))
    #         # here we make the sign of reward negative since the solver is minimizing
    #         # which is equivalent to maximizing the positive reward
    #         self.prog.AddCost(-1 * self.r[i])
