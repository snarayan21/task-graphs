import numpy as np
from scipy.stats import norm
import networkx as nx
from reward_model import RewardModel


class RewardModelEstimate(RewardModel):

    def __init__(self, num_tasks, num_robots, task_graph, coalition_params, coalition_types, dependency_params,
                 dependency_types, influence_agg_func_types):
        super().__init__(num_tasks, num_robots, task_graph, coalition_params, coalition_types, dependency_params,
                         dependency_types, influence_agg_func_types)

    def update_coalition_params(self, data, mode='oracle'):
        """
        Simulates the "disturbance" by changing the reward curves directly
        :return:
        """

        if mode == 'oracle':
            self.coalition_params = data
        elif mode == 'dataset':
            raise(NotImplementedError)
        else:
            raise(NotImplementedError("mode must be oracle or dataset, and dataset isn't implemented yet"))

    def get_coalition_params(self):
        return self.coalition_params
