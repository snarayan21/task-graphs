import numpy as np
import matplotlib.pyplot as plt
import toml
import argparse
import pathlib
import json

def main():

    parser = argparse.ArgumentParser(description='Create a graph.')
    parser.add_argument('-f', default=None, help='Path to experiment json file')
    #parser.add_argument('-baseline', '-b', action='store_true', default=False, help='include -baseline flag to additionally solve graph with baseline optimizer')
    args = parser.parse_args()

    with open(args.f, "r") as json_file:
        all_data = json.load(json_file)

    dep_vars = ['pruned_rounded_baseline_reward', 'pruned_rounded_greedy_reward','minlp_reward',
                'baseline_solution_time','greedy_solution_time','minlp_solution_time']

    # create a list of dicts, one for each experiment, with aggregated data for each result category over all trials
    agg_data_dict_list = []

    # aggregate data from each trial into a dict of lists for each experiment
    for exp in all_data.keys():
        agg_data_dict = {}
        for var in dep_vars:
            var_list = []
            for trial in all_data[exp].keys():
                var_list.append(all_data[exp][trial]['results'][var])
            agg_data_dict[var] = var_list
        agg_data_dict_list.append(agg_data_dict)

    # retrieve list of independent variables
    ind_var = 'makespan_constraint'
    x_list = []
    for exp in all_data.keys():
        for trial in all_data[exp].keys():
            x_list.append(all_data[exp][trial]['args']['exp'][ind_var])
            break

    # create dict of y_data lists
    y_data_dict = {}
    for var_name in dep_vars:
        y_data_dict[(var_name+'_mean')] = []
        y_data_dict[(var_name+'_std')] = []

    for agg_data_dict in agg_data_dict_list:
        for var in dep_vars:
            y_data_dict[(var+'_mean')].append(np.mean(np.array(agg_data_dict[var],dtype='float')))
            y_data_dict[(var+'_std')].append(np.std(np.array(agg_data_dict[var],dtype='float')))

    num_plots = 2
    fig, axs = plt.subplots(num_plots,1,sharex=True,figsize=(6,4*num_plots))
    legend_list = ['FLOW', 'GREEDY', 'MINLP']
    colors = ['blue','green','red']
    ylims = [[0,30],[-.5,8]]
    for ax_id in range(num_plots):
        for d in range(3):
            mean = np.array(y_data_dict[(dep_vars[ax_id*3 + d] + '_mean')])
            #axs[ax_id].plot(x_list, mean, label=legend_list[d],color=colors[d])
            std = np.array(y_data_dict[(dep_vars[ax_id*3 + d] + '_std')])
            axs[ax_id].errorbar(x_list, mean, yerr=std, label=legend_list[d],color=colors[d])
            #axs[ax_id].fill_between(x_list, mean+std, mean-std, color=colors[d], alpha=0.2)
            axs[ax_id].set_ylim(ylims[ax_id])
            axs[ax_id].legend()

    plt.show()
if __name__ == '__main__':
    main()
