import numpy as np
import matplotlib.pyplot as plt
import toml
import argparse
import pathlib
import json

def main():

    SMALL_SIZE = 10
    MEDIUM_SIZE = 15
    BIGGER_SIZE = 18
    BIGGEST_SIZE = 25

    #### FONT SIZE ###########################################################
    plt.rc('font', size=MEDIUM_SIZE) # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE) # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE) # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE) # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE) # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE) # legend fontsize
    plt.rc('figure', titlesize=BIGGEST_SIZE) # fontsize of the figure title

    parser = argparse.ArgumentParser(description='Create a graph.')
    parser.add_argument('-f', default=None, help='Path to experiment json file')
    parser.add_argument('-g', default=None, help="Type of experiment to graph -- makespan, n_tasks, or n_agents ")
    #parser.add_argument('-baseline', '-b', action='store_true', default=False, help='include -baseline flag to additionally solve graph with baseline optimizer')
    args = parser.parse_args()

    with open(args.f, "r") as json_file:
        all_data = json.load(json_file)

    dep_vars = ['pruned_rounded_baseline_reward', 'pruned_rounded_greedy_reward','minlp_reward',
                'baseline_solution_time','greedy_solution_time','minlp_solution_time']

    # create a list of dicts, one for each experiment, with aggregated data for each result category over all trials
    agg_data_dict_list = []
    norm_method = 'none' # 'minlp_reward' 'dual' 'greedy' or 'none'


    # aggregate data from each trial into a dict of lists for each experiment
    trials_to_delete = []
    trials_minlp_failed = [0 for _ in range(len(all_data.keys()))]
    trials_minlp_failed_list = []
    exp_minlp_run = [True for _ in range(len(all_data.keys()))]
    exp_ind = 0
    for exp in all_data.keys():
        for trial in all_data[exp].keys():
            print('EXP ', exp, ' TRIAL ', trial)
            for var in dep_vars:
                print(var, " ---- ", all_data[exp][trial]['results'][var])

    for exp in all_data.keys():
        agg_data_dict = {}
        try:
            all_data[exp]['trial 0']['args']['exp'].keys()
        except:
            breakpoint()
        if 'run_minlp' in all_data[exp]['trial 0']['args']['exp'].keys():
            exp_minlp_run[exp_ind] = all_data[exp]['trial 0']['args']['exp']['run_minlp']
        else:
            exp_minlp_run[exp_ind] = True
        for var in dep_vars:
            var_list = []
            for trial in all_data[exp].keys():
                if (not (np.array(all_data[exp][trial]['results']['pruned_rounded_baseline_solution'],dtype='float') == 0.0).all()) and \
                        (float(all_data[exp][trial]['results']['minlp_reward']) > 0.001 or not exp_minlp_run[exp_ind]) and \
                        np.abs(float(all_data[exp][trial]['results']['pruned_rounded_baseline_reward']))<10e10:
                    in_data = float(all_data[exp][trial]['results'][var])
                    if norm_method == 'minlp_reward':
                        if float(all_data[exp][trial]['results']['minlp_reward']) == 0:
                            norm_data = 1000000
                            print("ZERO NORM ON ", exp, trial)
                        else:
                            norm_data = in_data/float(all_data[exp][trial]['results']['minlp_reward'])
                    elif norm_method == 'dual':
                        if float(all_data[exp][trial]['results']['minlp_dual_bound']) == 0:
                            norm_data = 1000000
                            print("ZERO NORM ON ", exp, trial)
                        else:
                            norm_data = in_data/float(all_data[exp][trial]['results']['minlp_dual_bound'])
                    elif norm_method == 'greedy':
                        if float(all_data[exp][trial]['results']['pruned_rounded_greedy_reward']) == 0 or np.isnan(float(all_data[exp][trial]['results']['pruned_rounded_greedy_reward'])):
                            norm_data = 1000000
                            print("ZERO NORM ON ", exp, trial)
                        else:
                            norm_data = in_data/float(all_data[exp][trial]['results']['pruned_rounded_greedy_reward'])
                    elif norm_method == 'none':
                        if args.g == 'warm_start':
                            if var == 'minlp_reward':
                                norm_data = in_data - float(all_data[exp][trial]['results']['pruned_rounded_baseline_reward'])
                                if abs(norm_data) < 0.01:
                                    print("EXP ", exp, ", TRIAL ", trial, ". MINLP == FLOW")
                            else:
                                norm_data = in_data
                                if var == 'minlp_solution_time' and norm_data < 600:
                                    print("EXP ", exp, ", TRIAL ", trial, ". MINLP CONVERGED: ", all_data[exp][trial]['args']['exp']['warm_start'])
                        else:
                            norm_data = in_data
                    else:
                        raise(NotImplementedError("norm method must be 'minlp_reward', 'dual', or 'none'"))
                    if 'time' in var:
                        var_list.append(in_data)
                    else:
                        var_list.append(norm_data)
                elif float(all_data[exp][trial]['results']['minlp_reward']) < 0.001:
                    if not (exp, trial) in trials_minlp_failed_list:
                        print("MINLP FAILED TO COMPLETE EXP ", exp, " TRIAL ", trial)
                        #trials_to_delete.append((exp,trial))
                        trials_minlp_failed[exp_ind] = trials_minlp_failed[exp_ind] + 1
                        trials_minlp_failed_list.append((exp,trial))
                else:
                    if not (exp, trial) in trials_to_delete:
                        print("REMOVING EXP ", exp, " TRIAL ", trial, " no tasks were completed")
                        print("MINLP makespan: ", float(all_data[exp][trial]['results']['minlp_makespan']), "|| baseline makespan: ", float(all_data[exp][trial]['results']['pruned_baseline_makespan']))
                        trials_to_delete.append((exp,trial))
            agg_data_dict[var] = var_list
        agg_data_dict_list.append(agg_data_dict)
        exp_ind += 1

    for t in trials_to_delete:
        all_data[t[0]].pop(t[1])

    # retrieve list of independent variables
    if args.g=='makespan':
        ind_var = 'makespan_constraint'
    elif args.g=='n_tasks':
        ind_var = 'num_tasks'
    elif args.g=='n_agents':
        ind_var = 'num_robots'
    else:
        ind_var = args.g
    makespan = 0
    x_list = []
    for exp in all_data.keys():
        for trial in all_data[exp].keys():
            x_list.append(all_data[exp][trial]['args']['exp'][ind_var])
            makespan = all_data[exp][trial]['args']['exp']['makespan_constraint']
            break

    # create dict of y_data lists
    y_data_dict = {}
    for var_name in dep_vars:
        y_data_dict[(var_name+'_mean')] = []
        y_data_dict[(var_name+'_std')] = []
        y_data_dict[(var_name+'_all')] = []
    for agg_data_dict in agg_data_dict_list:
        for var in dep_vars:
            y_data_dict[(var+'_mean')].append(np.mean(np.array(agg_data_dict[var],dtype='float')))
            for i in range(len(y_data_dict[(var+'_mean')])):
                if not np.isfinite(y_data_dict[(var+'_mean')][i]) or np.isnan(y_data_dict[(var+'_mean')][i]):
                    y_data_dict[(var+'_mean')][i] = -1000000000000.0
            y_data_dict[(var+'_std')].append(np.std(np.array(agg_data_dict[var],dtype='float')))
            for i in range(len(y_data_dict[(var+'_std')])):
                if not np.isfinite(y_data_dict[(var+'_std')][i]) or np.isnan(y_data_dict[(var+'_std')][i]):
                    y_data_dict[(var+'_std')][i] = -1000000000000.0
            y_data_dict[(var+'_all')].append(np.array(agg_data_dict[var],dtype='float'))
            for i in range(len(y_data_dict[(var+'_all')])):
                for j in range(len(y_data_dict[(var+'_all')][i])):
                    if not np.isfinite(j) or np.isnan(j):
                        y_data_dict[(var+'_all')][i][j] = -1000000000000.0



    num_plots = 2
    if args.g=='makespan':
        fig, axs = plt.subplots(1,num_plots,figsize=(4*num_plots,4))
        ind_var = 'makespan_constraint'
        titles = ['Reward (Normalized) vs. Makespan Constraint', 'Computation time vs. Makespan Constraint']
        x_label = 'Makespan Constraint'
        xticks = [0.2,0.4,0.6,0.8,1.0]


    elif args.g=='n_tasks':
        ind_var = 'num_tasks'
        fig, axs = plt.subplots(num_plots,1,figsize=(4,num_plots*4)) #with 4.8 for right fig, 4 for left fig
        #titles = ['Reward (Normalized) vs. Number of Tasks', 'Computation time vs. Number of Tasks']
        #titles = ['Rel. Performance: Limited Domain', '']
        titles = ['Full Domain', '']
        x_label = 'Number of Tasks'
        #xticks = [6,8,10,15,20]
        #xticks = [6,8,10,15,20,25,30,35,40]
        xticks = [8,12,16,20]
        plt.gcf().subplots_adjust(left=0.2)


    elif args.g=='n_agents':
        ind_var = 'num_robots'
        fig, axs = plt.subplots(1,num_plots,figsize=(4*num_plots,4))
        titles = ['Reward vs. No. Agents', 'Comp. Time vs. No. Agents']
        x_label = 'Number of Agents'
        xticks = [2,4,6,10,15,20]
        plt.gcf().subplots_adjust(bottom=0.2,wspace=0.4)

    else:
         fig, axs = plt.subplots(1,num_plots,figsize=(4*num_plots,4))
         titles = [('Reward vs. ' + ind_var), ('Comp. time vs. ' + ind_var)]
         x_label = ind_var
         xticks = None


    legend_list = ['Flow: NLP', 'Flow: Greedy', 'MINLP']
    linestyles = ['-', '--', '-.']
    colors = ['blue','green','red']
    #titles = [(ind_var + ' vs reward'), (ind_var + ' vs computation time (s)')]
    y_labels = ['Reward', 'Computation time (s)']
    for ax_id in range(num_plots):
        max_mean = 0.0
        for d in range(3):
            plot_mask = [True for _ in range(len(x_list))]
            if d == 2:
                plot_mask = exp_minlp_run
            mean = np.array(y_data_dict[(dep_vars[ax_id*3 + d] + '_mean')])[plot_mask]
            if np.max(mean) > max_mean:
                max_mean = np.max(mean)
            axs[ax_id].plot(x_list, mean, label=legend_list[d],color=colors[d])
            std = np.array(y_data_dict[(dep_vars[ax_id*3 + d] + '_std')])[plot_mask]
            axs[ax_id].errorbar(np.array(x_list)[plot_mask], mean, yerr=std, color=colors[d],elinewidth=1.5, capsize=3, linestyle=linestyles[d],linewidth=3)

            scatter = True
            if scatter:
                x_list_ext = []
                y_list_ext = []
                for j in range(len(x_list)):
                    x_list_ext.extend([x_list[j] for _ in range(len(y_data_dict[(dep_vars[ax_id*3 + d] + '_all')][j]))])
                    y_list_ext.extend(y_data_dict[(dep_vars[ax_id*3 + d] + '_all')][j])
                kk = 0
                while kk < (len(x_list_ext)):
                    if y_list_ext[kk] > 5000 or np.isnan(y_list_ext[kk]) or np.isinf(y_list_ext[kk]):
                        y_list_ext.pop(kk)
                        x_list_ext.pop(kk)
                    else:
                        kk = kk + 1
                axs[ax_id].scatter(x_list_ext, y_list_ext, label=legend_list[d], color=colors[d])
                draw_fit_line = True
                if draw_fit_line:
                    # ++++++++++++ FIT LINES +++++++++++++
                    fit_line = np.poly1d(np.polyfit(x_list_ext,y_list_ext,1))
                    fit_x = np.linspace(np.min(x_list_ext), np.max(x_list_ext))
                    axs[ax_id].plot(fit_x,fit_line(fit_x),color=colors[d])
                    #axs[ax_id].fill_between(x_list, mean+std, mean-std, color=colors[d], alpha=0.2)
                print("ALL DATA ", legend_list[d])
                print(x_list_ext)
                print(y_list_ext)
            if xticks is not None:
                axs[ax_id].set_xticks(xticks)
            axs[ax_id].legend()
            axs[ax_id].set_ylabel(y_labels[ax_id])
            axs[ax_id].set_xlabel(x_label)
            axs[ax_id].title.set_text(titles[ax_id])
        # for k in range(len(all_data.keys())):
        #     axs[ax_id].text(x_list[k],y_data_dict[(dep_vars[ax_id*3] + '_mean')][k]*1.05,str(trials_minlp_failed[k]))
        ylims = [[0,1.1*max_mean],[-.5,1.1*max_mean]]
        #ylims = [[0,300],[-.5,1.1*max_mean]]
        axs[ax_id].set_ylim(ylims[ax_id])

    plt.show()
    #create output data for heatmap
    output = {}
    output['x'] = x_list
    output['NLP_mean'] = y_data_dict['pruned_rounded_baseline_reward_mean']
    output['NLP_std'] = y_data_dict['pruned_rounded_baseline_reward_std']
    output['NLP_comp_time_mean'] = y_data_dict['baseline_solution_time_mean']
    output['NLP_comp_time_std'] = y_data_dict['baseline_solution_time_std']

    output['greedy_mean'] = y_data_dict['pruned_rounded_greedy_reward_mean']
    output['greedy_std'] = y_data_dict['pruned_rounded_greedy_reward_std']
    output['greedy_comp_time_mean'] = y_data_dict['greedy_solution_time_mean']
    output['greedy_comp_time_std'] = y_data_dict['greedy_solution_time_std']

    output['MINLP_mean'] = y_data_dict['minlp_reward_mean']
    output['MINLP_std'] = y_data_dict['minlp_reward_std']
    output['MINLP_time_mean'] = y_data_dict['minlp_solution_time_mean']
    output['MINLP_time_std'] = y_data_dict['minlp_solution_time_std']

    output['makespan'] = makespan
    with open(("heatmap_data/"+args.f),"w") as outfile:
        json.dump(output, outfile)
if __name__ == '__main__':
    main()
