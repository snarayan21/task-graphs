import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

SMALL_SIZE = 10
MEDIUM_SIZE = 20
BIGGER_SIZE = 32
#### FONT SIZE ###########################################################
plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# FONT #################################
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.usetex"] = True
plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}', r'\usepackage{bm}']


def flow_reward_plot(max_steps, flow_no_failure,
                     total_reward_no_failure,
                     flow_failure,
                     total_reward_failure):
    """
    Plots the comparison among the flow and rewards
    """
    # plot the figures
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10.5, 8.5)
    ax.set_ylim([-0.1, 10.0])
    # for j in range(num_targets):
    ax.plot(0.033 * np.arange(max_steps), total_reward_no_failure, linestyle='-', color='red', linewidth=4,
            # palette[j]
            label='Not Adaptive')
    ax.plot(0.033 * np.arange(max_steps), total_reward_failure, linestyle='--', color='black', linewidth=4,
            label='Adaptive')
    ax.set_xlabel('Time')
    ax.set_ylabel(r'Total Task Reward')
    ax.legend(loc='upper right')
    # fig.tight_layout()
    plt.show()