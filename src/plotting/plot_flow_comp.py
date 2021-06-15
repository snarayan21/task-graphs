import pickle as pkl
import matplotlib.pyplot as plt
from pathlib import Path
import toml
import sys
sys.path.append("../../include")
from plot_utils import *
import numpy as np

task_root_dir = Path(__file__).resolve().parents[2]
data_dir = task_root_dir / 'data'

experiment_file_path = str(data_dir) + '/' + 'farm_no_adaptive.pkl'
with open(experiment_file_path, "rb") as f:
    data_farm_no_adaptive = pkl.load(f)

experiment_file_path = str(data_dir) + '/' + 'farm_adaptive.pkl'
with open(experiment_file_path, "rb") as f:
    data_farm_adaptive = pkl.load(f)
#
# import pdb
# pdb.set_trace()
flow_reward_plot(data_farm_no_adaptive['max_steps'], data_farm_no_adaptive["flow_store"],
                 np.sum(data_farm_no_adaptive["task_reward_store"], axis=1),
                 data_farm_adaptive["flow_store"],
                 np.sum(data_farm_adaptive["task_reward_store"], axis=1))
