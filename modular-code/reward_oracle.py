import numpy as np

class RewardOracle():

    def __init__(self, mean_func, variance_func, reward_func, influence_agg_func_type = 'm',
                 name='no_name', node_id=-1):
        #may want to add in influence func and delta func later, to reason about how these impact reward distribution
        self.mean_func = mean_func
        self.variance_func = variance_func
        self.reward_func = reward_func
        self.influence_agg_func = self.get_influence_agg_func(influence_agg_func_type)
        self.name = name
        self.node_id = node_id



    def sample_reward(self, rho, deltas):
        """ Samples the reward pdf given a coalition and an influence function output
        :arg rho is currently a scalar integer representing the coalition (i.e. the number of robots, in this
         homogeneous case). This should be replaced with a coalition function output, or perhaps the coalition vector
        :arg deltas is a list of the influencing nodes' influence function outputs
        """
        agg_delta = self.influence_agg_func(deltas)
        reward_func_val = self.reward_func(rho,agg_delta)
        mean = self.mean_func(reward_func_val)
        var = self.variance_func(reward_func_val)
        return np.random.normal(loc=mean, spread=np.sqrt(var))


    def get_mean_var(self, rho, deltas):
        """ Gets the mean and variance of the reward pdf given a coalition and an influence function output
        :arg rho is currently a scalar integer representing the coalition (i.e. the number of robots, in this
         homogeneous case). This should be replaced with a coalition function output, or perhaps the coalition vector
        :arg deltas is a list of the influencing nodes' influence function outputs
        :return: (mean, var)
        """
        agg_delta = self.influence_agg_func(deltas)
        reward_func_val = self.reward_func(rho,agg_delta)
        mean = self.mean_func(reward_func_val)
        var = self.variance_func(reward_func_val)

        return mean, var

    def get_influence_agg_func(self, influence_agg_func_type):
        if influence_agg_func_type == 'm':
            def mult_agg(influence_func_output_list):
                influence_aggregated = 1
                for i in range(len(influence_func_output_list)):
                    influence_aggregated = influence_aggregated*influence_func_output_list[i]
                return influence_aggregated
            return mult_agg
        else:
            raise NotImplementedError('Influence aggregation type ' + influence_agg_func_type + ' is not supported.' )
