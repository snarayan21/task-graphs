import numpy as np

class RewardOracle():

    def __init__(self, mean_func, variance_func, reward_func,
                 name='no_name', node_id=-1):
        #may want to add in influence func and delta func later, to reason about how these impact reward distribution
        self.mean_func = mean_func
        self.variance_func = variance_func
        self.reward_func = reward_func
        self.name = name
        self.node_id = node_id



    def sample_reward(self, rho, delta):
        """ Samples the reward pdf given a coalition and an influence function output
        :arg rho is currently a scalar integer representing the coalition (i.e. the number of robots, in this
         homogeneous case). This should be replaced with a coalition function output, or perhaps the coalition vector
        :arg delta is the influence function output, aggregated over all influencing nodes
        """
        reward_func_val = self.reward_func(rho,delta)
        mean = self.mean_func(reward_func_val)
        var = self.variance_func(reward_func_val)
        return np.random.normal(loc=mean, spread=np.sqrt(var))


    def get_mean_var(self, rho, delta):
        """ Gets the mean and variance of the reward pdf given a coalition and an influence function output
        :arg rho is currently a scalar integer representing the coalition (i.e. the number of robots, in this
         homogeneous case). This should be replaced with a coalition function output, or perhaps the coalition vector
        :arg delta is the influence function output, aggregated over all influencing nodes
        :return: (mean, var)
        """

        reward_func_val = self.reward_func(rho,delta)
        mean = self.mean_func(reward_func_val)
        var = self.variance_func(reward_func_val)

        return mean, var
