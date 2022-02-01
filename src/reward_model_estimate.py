import numpy as np
from scipy.stats import norm
import networkx as nx
from reward_model import RewardModel


class RewardModelEstimate(RewardModel):
    """
    RewardModelEstimate is a subclass of RewardModel - it models the rewards of a task graph and is initialized with
    the graph and reward parameters. This subclass adds functionality to update the reward model based on data.
    """

    def __init__(self, num_tasks, num_robots, task_graph, coalition_params, coalition_types, dependency_params,
                 dependency_types, influence_agg_func_types):
        super().__init__(num_tasks, num_robots, task_graph, coalition_params, coalition_types, dependency_params,
                         dependency_types, influence_agg_func_types)

    def update_coalition_params(self, data, mode='oracle'):
        """
        :arg data: the data to use to update the coalition parameters. Different type depending on mode: when mode is
        "oracle" it should be an n_nodes long list of coalition param lists for each node.
        :arg mode: the type of coalition parameter update. "oracle" (default) updates all coalition func
        parameters with the values passed into data. "dataset" mode is passing in flow-reward dataset information, which
        is used to update the parameters (not yet implemented)
        :return: updated coalition_params
        """

        if mode == 'oracle':
            self.coalition_params = data
        elif mode == 'dataset':
            raise(NotImplementedError)
        else:
            raise(NotImplementedError("mode must be oracle or dataset, and dataset isn't implemented yet"))

        return self.coalition_params

    def get_coalition_params(self):
        return self.coalition_params
