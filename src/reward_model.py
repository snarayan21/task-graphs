import numpy as np
from scipy.stats import norm
from pydrake.autodiffutils import AutoDiffXd
import networkx as nx
import autograd
import math


class RewardModel:
    """
    A class that stores and computes the reward over a task graph given a robot flow. The model is initialized with
    reward and graph parameters, and retains these parameters throughout computation. It is the "real-world" model of
    rewards.
    """

    def __init__(self, num_tasks, num_robots, task_graph, coalition_params, coalition_types, dependency_params,
                 dependency_types, influence_agg_func_types):
        self.num_tasks = num_tasks
        self.num_robots = num_robots
        self.task_graph = task_graph
        self.edges = [list(edge) for edge in self.task_graph.edges]
        #breakpoint()
        self.num_edges = len(self.edges)
        self.incidence_mat = nx.linalg.graphmatrix.incidence_matrix(self.task_graph,
                                                                    oriented=True).A  # TODO this duplicates a line in initializeSolver, should fix?
        self.adjacency_mat = nx.linalg.graphmatrix.adjacency_matrix(self.task_graph).A

        self.coalition_params = coalition_params
        self.coalition_types = coalition_types
        self.dependency_params = dependency_params
        self.dependency_types = dependency_types
        self.influence_agg_func_types = influence_agg_func_types

    def flow_cost(self, f):
        """
        Computes the cost function value over the entire task graph
        :return:
        """
        #breakpoint()
        return np.sum(self._nodewise_optim_cost_function(f))

    def _nodewise_optim_cost_function(self, f, eval=False, use_cvar=False):
        """
        Computes the cost function value for all the nodes individually based on the flow value
        :param f: shape=(num_edges X 1) , flow value over all edges

        :return: shape=(num_tasks X 1) , cost for each node
        """
        # total incoming flow into each node
        incoming_flow = self._compute_incoming_flow(f)

        var_reward_mean = np.zeros(self.num_tasks, dtype=object)
        var_reward_stddev = np.zeros(self.num_tasks, dtype=object)
        var_reward = np.zeros(self.num_tasks, dtype=object)

        node_cost_val = np.zeros(self.num_tasks, dtype=object)

        for node_i in range(self.num_tasks):
            # Compute Coalition Function
            node_coalition = self._compute_node_coalition(node_i, incoming_flow[node_i])
            # Compute the reward by combining with Inter-Task Dependency Function
            # influencing nodes of node i

            if eval:
                var_reward_mean[node_i], var_reward_stddev[node_i] = self.compute_node_reward_dist(node_i,
                                                                                                     node_coalition,
                                                                                                     var_reward,
                                                                                                     var_reward_stddev)
                var_reward[node_i] = np.random.normal(var_reward_mean[node_i], var_reward_stddev[node_i])

            else:
                var_reward_mean[node_i], var_reward_stddev[node_i] = self.compute_node_reward_dist(node_i,
                                                                                                     node_coalition,
                                                                                                     var_reward_mean,
                                                                                                     var_reward_stddev)
            #breakpoint()
            if use_cvar:
                # if use_cvar is True, use the cvar metric to compute the cost
                node_cost_val[node_i] = self._cvar_cost(var_reward_mean[node_i],
                                                        var_reward_stddev[node_i])
            else: # TODO should we return the means or the samples?
                node_cost_val[node_i] = var_reward_mean[node_i]
        if eval:
            return var_reward
        # return task-wise cost (used in optimization)
        return -node_cost_val

    def get_mean_std(self, node_i, rho, deltas):
        """ Gets the mean and std deviation of the reward pdf given a coalition and an influence function output
        :arg rho is currently a scalar integer representing the coalition (i.e. the number of robots, in this
         homogeneous case). This should be replaced with a coalition function output, or perhaps the coalition vector
        :arg deltas is a list of the influencing nodes' influence function outputs
        :return: (mean, std)
        """
        influence_agg_func = getattr(self, 'influence_agg_' + self.influence_agg_func_types[node_i])

        def std_dev_func(val):
            return 0.1 * val

        agg_delta = influence_agg_func(deltas)
        reward_func_val = agg_delta + rho
        mean = reward_func_val
        std = std_dev_func(reward_func_val)
        return mean, std

    def _cvar_cost(self, mean, std):
        """

        :param mean:
        :param std:
        :return:
        """
        if not hasattr(self, 'cvar_coeff'):
            # TODO should this be negative? should this look at the lower tail of the distribution instead of the upper??
            alpha = 0.05
            inv_cdf = norm.ppf(alpha)
            numerator = norm.pdf(inv_cdf)
            self.cvar_coeff = - numerator / (1 - alpha)
        cvar_cost = mean + std * self.cvar_coeff
        return cvar_cost

    def get_dynamics_equations(self):
        """

        :return: function handle that takes in (x[l], u[l], l) and returns x[l+1]
        """
        def dynamics(x, u, node_i):
            """
            :arg x is the vector of rewards at the incoming neighborhood of node node_i.
            :arg u is the vector of flows along the incoming edges to node node_i.
            :arg node_i is the index of the node
            """
            if np.isscalar(u):
                sum_u = u
            else:
                sum_u = sum(u)
            node_coalition = self._compute_node_coalition(node_i, sum_u)
            reward_mean, reward_std = self.compute_node_reward_dist(node_i, node_coalition, x, 0)
            return reward_mean

        def dynamics_b(x, u, node_i, additional_x, l_index):
            """
            :arg x is the reward at node l, the
            :arg u is the vector of flows along the incoming edges to node node_i.
            :arg node_i is the index of the node
            """
            x = np.atleast_1d(x)
            additional_x = np.atleast_1d(additional_x)
            # assemble x vector
            if (not additional_x.size == 0) and l_index != -1:
                #full_x = np.insert(additional_x,l_index,x)
                full_x = np.concatenate((additional_x[0:l_index],x,additional_x[l_index:len(additional_x)]))
            elif l_index == -1:
                full_x = additional_x
            else:
                full_x = x
            if np.isscalar(u):
                sum_u = u
            else:
                sum_u = sum(u)
            node_coalition = self._compute_node_coalition(node_i, sum_u)
            #breakpoint()
            reward_mean, reward_std = self.compute_node_reward_dist(node_i, node_coalition, full_x, 0)
            return reward_mean

        #breakpoint()
        return dynamics_b

    def compute_node_reward_dist(self, node_i, node_coalition, reward_mean, reward_std):
        """
        For a given node, this function outputs the mean and std dev of the reward based on the coalition function
        of the node, the reward means of influencing nodes, and the corresponding task influence functions
        TODO: use reward std of previous nodes in computation as well?
        :param node_i: shape=(1x1), index of node for which coalition function has to be evaluated
        :param node_coalition: shape=(1x1),  coalition value for the node
        :param reward_mean: shape=(num_tasks x 1), mean of rewards for all tasks (only partially filled)
        :param reward_std: shape=(num_tasks x 1), std dev of rewards for all tasks (only partially filled)
        :return: (mean, std) - two scalars
        """
        # compute incoming edges to node_i
        incoming_edges = list(self.task_graph.in_edges(node_i))
<<<<<<< HEAD:modular-code/reward_model.py
        print("incoming edges:", incoming_edges)
        print("Computing node reward for NODE ",node_i, " with an incoming coalition of size ", node_coalition)
=======
        #print("Computing node reward for NODE ",node_i, " with an incoming coalition of size ", node_coalition)
>>>>>>> origin/main:src/reward_model.py
        task_influence_value = []
        list_ind = 0
        for edge in incoming_edges:
            # find global index of edge
            edge_id = list(self.task_graph.edges).index(edge)
            # find the source node of that edge
            source_node = self.edges[edge_id][0]
            # extract the delta function applicable to this edge
            task_interdep = getattr(self, self.dependency_types[edge_id])
            print("edge id: ", edge_id)
            print("task influence function:", task_interdep)
            # compute the task influence value (delta for an edge). if "null" then
            if task_interdep.__name__ != 'null':

                if np.isscalar(reward_mean) or isinstance(reward_mean, autograd.numpy.numpy_boxes.ArrayBox):
                    task_influence_value.append(task_interdep(reward_mean,
                                                              self.dependency_params[edge_id]))
                else:
                    # we passed in a list of only incoming edges flow
                    task_influence_value.append(task_interdep(reward_mean[list_ind],
                                                              self.dependency_params[edge_id]))
                    list_ind += 1
        mean, std = self.get_mean_std(node_i, node_coalition, task_influence_value)
        #breakpoint()
        return mean, std

    def _compute_node_coalition(self, node_i, f):
        """
        This function computes the coalition function output for a given node and a given flow
        :param node_i, shape=(1x1), index of node for which coalition function has to be evaluated
        :param f, shape=(1x1), flow incoming into the node
        :return: shape=(1x1), coalition function output (0 if source node)
        """
        if node_i != 0 or node_i != self.num_tasks:
            coalition_function = getattr(self, self.coalition_types[node_i])
            #breakpoint()
            return coalition_function(f, param=self.coalition_params[node_i])
        else:
            # source and sink node has 0 coalition/reward
            return 0

    def _compute_incoming_flow(self, f):
        """
        Computes the total incoming flow into each node
        :return:
        """
        D_incoming = np.maximum(self.incidence_mat, 0)
        # total incoming flow into each node
        return D_incoming @ f

    def sigmoid(self, flow, param):
        if type(flow) is AutoDiffXd:
            return param[0] / (1 + np.exp(-1 * param[1] * (flow - param[2])))
        return param[0] / (1 + math.e ** (-1 * param[1] * (flow - param[2])))

    def dim_return(self, flow, param):
        return param[0] - param[2] * np.exp(-1 * param[1] * flow)
        # return param[0] + (param[2] * (1 - np.exp(-1 * param[1] * flow)))

    def polynomial(self, flow, param):
        val = 0
        for i in range(len(param)):
            val += param[i]*flow**i
            #print('poly ', i, flow, val)
        return val

    def null(self, flow, param):
        """
        :param flow:
        :return:
        """
        return 0.0

    def influence_agg_and(self, deltas):
        return np.prod(np.array(deltas))

    def influence_agg_or(self, deltas):
        #print('or ', deltas, np.sum(np.array(deltas)))
        return np.sum(np.array(deltas))

    def compare_func(self, func):
        for l in range(3):
            print('l = ', l)
            for i in range(10):
                print(func(np.array([0])[0],np.array([i])[0],l))
