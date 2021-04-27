

class RewardOracle():

    def __init__(self, coalition_func, coalition_func_class, influence_func, influence_func_class, gaussian):
        pass

    def sample_reward(self, rho, delta):
        """ Samples the reward pdf given a coalition and an influence function output
        :arg rho is the coalition function output, or perhaps the coalition vector
        :arg delta is the influence function output, aggregated over all influencing nodes
        """
        pass

    def get_mean_var(self, rho, delta):
        """ Gets the mean and variance of the reward pdf given a coalition and an influence function output
        :arg rho is the coalition function output, or perhaps the coalition vector
        :arg delta is the influence function output, aggregated over all influencing nodes
        :return: (mean, var)
        """
