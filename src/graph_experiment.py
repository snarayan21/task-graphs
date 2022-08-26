import numpy as np
import matplotlib.pyplot as plt
import toml
import argparse
import pathlib


def main():

    parser = argparse.ArgumentParser(description='Create a graph.')
    parser.add_argument('-f', '--names_list', nargs='+', default=[], help='Specify paths to the experiments to be graphed')
    #parser.add_argument('-baseline', '-b', action='store_true', default=False, help='include -baseline flag to additionally solve graph with baseline optimizer')
    args = parser.parse_args()

    path_list = [pathlib.Path(filename) for filename in args.names_list]
    args_list = []
    results_list_list = []
    for path in path_list:
        results_list = []
        for trial_dir in path.iterdir():
            if trial_dir.is_dir():
                print('loading ', trial_dir.absolute())
                results_list.append(toml.load((trial_dir / 'results.toml').absolute()))
        results_list_list.append(results_list)
        args_path = path / 'trial_0/args.toml'
        print('loading ', args_path.absolute())
        args = toml.load(args_path.absolute())
        args_list.append(args['exp']['makespan_constraint'])


    baseline_pruned_rounded_list_list = []
    greedy_pruned_rounded_list_list = []
    minlp_list_list = []
    baseline_time_list_list = []
    greedy_time_list_list = []
    minlp_time_list_list = []

    data_struct = np.zeros((6*2,len(results_list_list)))
    for results_list in results_list_list:
        baseline_pruned_rounded_list = []
        greedy_pruned_rounded_list = []
        minlp_list = []
        baseline_time_list = []
        greedy_time_list = []
        minlp_time_list = []
        for result in results_list:
            baseline_pruned_rounded_list.append(result['pruned_rounded_baseline_reward'])
            greedy_pruned_rounded_list.append(result['pruned_rounded_greedy_reward'])
            minlp_list.append(result['minlp_reward'])
            baseline_time_list.append(result['baseline_solution_time'])
            greedy_time_list.append(result['greedy_solution_time'])
            minlp_time_list.append(result['minlp_solution_time'])

        baseline_pruned_rounded_list_list.append(baseline_pruned_rounded_list)
        greedy_pruned_rounded_list_list.append(greedy_pruned_rounded_list)
        minlp_list_list.append(minlp_list)
        baseline_time_list_list.append(baseline_time_list)
        greedy_time_list_list.append(greedy_time_list)
        minlp_time_list_list.append(minlp_time_list)

    data_list_list_list = [baseline_pruned_rounded_list_list, greedy_pruned_rounded_list_list, minlp_list_list,
                           baseline_time_list_list, greedy_time_list_list, minlp_time_list_list]

    for data_pt in range(len(results_list_list)):
        for i in range(len(data_list_list_list)):
            data_struct[2*i,data_pt] = np.mean(np.array(data_list_list_list[i][data_pt], dtype='float'))
            data_struct[2*i+1,data_pt] = np.std(np.array(data_list_list_list[i][data_pt], dtype='float'))
    breakpoint()
    fig, axs = plt.subplots(2,1,sharex=True,figsize=(6,9))
    legend_list = ['FLOW', 'GREEDY', 'MINLP']
    ylims = [[0,30],[-.5,8]]
    for ax_id in range(len(axs)):
        for d in range(3):
            axs[ax_id].plot(args_list,data_struct[(ax_id)*6 + 2*d,:], label=legend_list[d])
            axs[ax_id].set_ylim(ylims[ax_id])
            axs[ax_id].legend()

    plt.show()
if __name__ == '__main__':
    main()
