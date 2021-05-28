import numpy as np
from scipy.stats import norm
import networkx as nx


class RewardModel:

    def __init__(self, num_tasks, num_robots, edges, task_graph, coalition_params, coalition_types, dependency_params,
                 dependency_types, influence_agg_func_types):
        self.num_tasks = num_tasks
        self.num_robots = num_robots
        self.edges = edges
        self.num_edges = len(self.edges)
        self.task_graph = task_graph
        self.incidence_mat = nx.linalg.graphmatrix.incidence_matrix(self.task_graph,
                                                                    oriented=True).A  # TODO this duplicates a line in initializeSolver, should fix?


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
        return np.sum(self._nodewise_optim_cost_function(f))

    def _nodewise_optim_cost_function(self, f, eval=False):
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
            # use the cvar metric to compute the cost
            node_cost_val[node_i] = self._cvar_cost(var_reward_mean[node_i],
                                                    var_reward_stddev[node_i])
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
        reward_func_val = agg_delta * rho
        mean = reward_func_val
        std = std_dev_func(reward_func_val)

        return mean, std

    def get_influence_agg_func(self, influence_agg_func_type):
        if influence_agg_func_type == 'm':
            def mult_agg(influence_func_output_list):
                influence_aggregated = 1
                for i in range(len(influence_func_output_list)):
                    influence_aggregated = influence_aggregated * influence_func_output_list[i]
                return influence_aggregated

            return mult_agg
        else:
            raise NotImplementedError('Influence aggregation type ' + influence_agg_func_type + ' is not supported.')

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

        task_influence_value = []
        for edge in incoming_edges:
            # find global index of edge
            edge_id = list(self.task_graph.edges).index(edge)
            # find the source node of that edge
            source_node = self.edges[edge_id][0]
            # extract the delta function applicable to this edge
            task_interdep = getattr(self, self.dependency_types[edge_id])
            # compute the task influence value (delta for an edge). if "null" then
            if task_interdep.__name__ != 'null':
                task_influence_value.append(task_interdep(reward_mean[source_node],
                                                          self.dependency_params[edge_id]))

        mean, std = self.get_mean_std(node_i, node_coalition, task_influence_value)

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
        return param[0] / (1 + np.exp(-1 * param[1] * (flow - param[2])))

    def dim_return(self, flow, param):
        return param[0] - param[2] * np.exp(-1 * param[1] * flow)
        # return param[0] + (param[2] * (1 - np.exp(-1 * param[1] * flow)))

    def null(self, flow, param):
        """
        :param flow:
        :return:
        """
        return 0.0

    def influence_agg_and(self, deltas):
        return np.prod(np.array(deltas))

    def influence_agg_or(self, deltas):
        return np.sum(np.array(deltas))
