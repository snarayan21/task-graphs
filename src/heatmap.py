import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import json
import argparse

def main():

    parser = argparse.ArgumentParser(description='Generate heat map from experiment output json files')
    parser.add_argument('-f','--filelist', nargs='+', help='<Required> List of json files', required=True)
    args = parser.parse_args()

    input_dict_list = []
    for filename in args.filelist:
        with open(filename, "r") as loadfile:
            input_dict_list.append(json.load(loadfile))

    SMALL_SIZE = 10
    MEDIUM_SIZE = 15
    BIGGER_SIZE = 18
    BIGGEST_SIZE = 25

    #### FONT SIZE ###########################################################
    plt.rc('font', size=MEDIUM_SIZE) # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE) # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE) # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE) # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE) # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE) # legend fontsize
    plt.rc('figure', titlesize=BIGGEST_SIZE) # fontsize of the figure title

    # FONT #################################
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["text.usetex"] = True
    plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

    flow_rewards = []
    greedy_rewards = []
    makespans = []
    for d in input_dict_list:
        flow_rewards.append(d['flow'])
        greedy_rewards.append(d['greedy'])
        makespans.append(d['makespan'])

    sort_inds = np.flip(np.argsort(np.array(makespans, dtype='int')))
    flow_rewards_plot = np.array(flow_rewards)[sort_inds]
    greedy_rewards_plot = np.array(greedy_rewards)[sort_inds]

    fig, axs = plt.subplots(1, 3, gridspec_kw={'width_ratios':[1,1,0.08]})
    #fig.tight_layout(pad=0.5)
    plt.subplots_adjust(left=0.1, right=0.9,bottom=0.15,top=0.9, wspace=0.3)
    g1 = sns.heatmap(flow_rewards_plot, linewidth=0.5, cmap='RdBu', center=1, xticklabels=[8, 12, 16, 20], yticklabels=[1.0, 0.75, 0.50, 0.25], ax=axs[0], vmin=0, vmax=5, cbar_ax=axs[2], cbar_kws={"label":"Normalized reward: flow NLP"}, square=True)
    g2 = sns.heatmap(greedy_rewards_plot, linewidth=0.5, cmap='RdBu', cbar=False, center=1, xticklabels=[8, 12, 16, 20], yticklabels=[1.0, 0.75, 0.50, 0.25], ax=axs[1], vmin=0, vmax=5, square=True)# cbar_kws={"label":"%"}
    #g3 = sns.heatmap(abs_radius, cmap='RdBu', cbar=True, xticklabels=[],yticklabels=[], center=0, cbar_ax=axs[2])

    g1.set_xlabel('Number of Tasks')
    g1.set_ylabel('Makespan Constraint')
    g2.set_xlabel('Number of Tasks')
    g2.set_ylabel('Makespan Constraint')

    plt.show()

if __name__ == '__main__':
    main()
