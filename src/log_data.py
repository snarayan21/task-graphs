import numpy as np
import pickle as pkl
from pathlib import Path

class LogData:
    """
    A class for logging data from task planning experiments
    """

    def __init__(self, max_steps, n_agents, num_tasks, num_edges, scenario="test" ):

        self.max_steps = max_steps
        self.n_agents = n_agents
        self.scenario_name = scenario
        self.time_store = np.arange(self.max_steps)  # time vector
        self.flow_store = np.zeros((self.max_steps, num_edges))  #store flow values over time
        self.task_reward_store = np.zeros((self.max_steps, num_tasks))
        self.task_root_dir = Path(__file__).resolve().parents[1]
        self.data_dir = self.task_root_dir / 'data'

    def store_in_loop(self, time_step, flow, reward):
        """

        :param time_step:
        :param flow:
        :param reward:
        :return:
        """

        self.flow_store[time_step, :] = flow
        self.task_reward_store[time_step, :] = reward

    def write_to_file(self):

        data = vars(self) # stores the entire object into a dict
        pickle_file = str(self.data_dir)+'/'+self.scenario_name+".pkl"
        with open(pickle_file, "wb+") as f:
            pkl.dump(data, f)


