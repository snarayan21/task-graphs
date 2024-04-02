import traceback

import matplotlib.pyplot as plt

from real_time_solver import RealTimeSolver
import pathlib
import toml
import numpy as np
import argparse
from argparse import Namespace
from experiment_generator import ExperimentGenerator, clean_dir_name
import sys
from matplotlib import pyplot as plt
import time

# create new experiment args
parser = argparse.ArgumentParser(description='Run a real-time reallocation experiment')
parser.add_argument('-cfg', default=None, help='Specify path to the experiment toml file')
parser.add_argument('-outputpath', '-o', default=None, help='Base name of output experiment dir')
parser.add_argument('-inputargs', default=None, help='Do not use, included for compatibility')
parser.add_argument('-debug', default=False, action='store_true', help='Run trials in debug mode, output to terminal' )
parser.add_argument('-white_noise', default=None, help='Amount of white noise added to reward observations (specified as a ratio of std to mean)')
args = parser.parse_args()

experiment_generator_args = Namespace(
    cfg=args.cfg,
    outputpath=args.outputpath,
    inputargs=args.inputargs,
)

### DANGEROUS but these warnings are annoying ###
import warnings
warnings.filterwarnings("ignore")

# todo break this into sub arg dict specific to experiment generator
experiment_generator = ExperimentGenerator(args)

all_args = toml.load(args.cfg)
exp_args = all_args['exp']

experiment_dir = experiment_generator.experiment_dir

real_time_args_file = experiment_dir / "real_time_args.toml"
out_args_dict = {}
for key, value in vars(args).items():
    out_args_dict[key] = value
out_args_dict['cfg_file_contents'] = all_args

with open(real_time_args_file, "w") as f:
    toml.dump(out_args_dict, f)

perturbation = 'catastrophic'
if args.white_noise is None:
    perturbation_params = 0.15
else:
    perturbation_params = float(args.white_noise)
perturb_model = False # perturb model parameters to simulate model error
solve_greedy = False

num_trials = 20
num_tests_per_trial = 10
num_evals_per_trial = 50
draw_graphs = False
tests_per_noise_level = 50 # used if single_model_noise_test

single_model_noise_test = True
noise_levels = np.arange(0.0, 0.3, 0.05)
if single_model_noise_test:
    num_tests_per_trial = len(noise_levels)*tests_per_noise_level

# DATA LOGGING
original_rewards = [[] for _ in range(num_trials)]
real_time_rewards = [[] for _ in range(num_trials)]
greedy_rewards = [[] for _ in range(num_trials)]
all_trial_reward_perturbations_abs = [[] for _ in range(num_trials)]
all_trial_reward_perturbations_rel = [[] for _ in range(num_trials)]

total_c_perturbations = [[] for _ in range(num_trials)]
indirect_c_perturbations = [[] for _ in range(num_trials)]

start = time.time()

for trial in range(num_trials):
    trial_args, nx_task_graph, node_pos = experiment_generator.generate_taskgraph_args()
    #TODO for now, do not run MINLP solver. Eventually, may want to integrate
    trial_args['exp']['run_minlp'] = False
    dir_name = "trial_" + str(trial)
    trial_dir = experiment_dir / dir_name
    trial_dir.mkdir(parents=True, exist_ok=False)
    args_file = trial_dir / "args.toml"
    with open(args_file, "w") as f:
        toml.dump(trial_args,f)
    log_file = open(str(trial_dir / "log.txt"), "w")
    print(f"Changing log file to {str(trial_dir / 'log.txt')}")
    if not args.debug:
        sys.stdout = log_file
    for t in range(num_tests_per_trial):
        try:
            reward_perturbations_abs = []
            reward_perturbations_rel = []
            if single_model_noise_test:
                perturbation_params = noise_levels[int(np.floor(t/tests_per_noise_level))]
            real_time_solver = RealTimeSolver(trial_args['exp'], trial_dir=str(trial_dir), draw_graph=draw_graphs)
            solver_done = False
            while not solver_done:

                # TODO when integrating perturbations to reward_model parameters
                if perturb_model:
                    # TODO not yet implemented
                    solver_done, actual_reward, r_exp = real_time_solver.sim_step_perturbed_model(perturbation, perturbation_params, draw_graph=draw_graphs)
                else:
                    solver_done, actual_reward, r_exp = real_time_solver.sim_step(perturbation, perturbation_params, draw_graph=draw_graphs)

                reward_perturbations_abs.append(actual_reward - r_exp)
                reward_perturbations_rel.append((actual_reward - r_exp) / actual_reward)
                # TODO RESUME DEBUGGING USING BELOW
                # if abs((actual_reward - r_exp) / actual_reward)>20:
                #     import pdb; pdb.set_trace()

        except (Exception, OverflowError) as e:
            print(e)
            print(traceback.format_exc())
            original_rewards[trial].append(1000000000)
            real_time_rewards[trial].append(0.00001)
            total_c_perturbations[trial].append(0.0)
            indirect_c_perturbations[trial].append(0.0)
            if solve_greedy:
                greedy_rewards[trial].append(1000000000)
            continue
        if perturbation is None:
            original_rewards[trial] = np.sum(real_time_solver.original_rewards)
            if solve_greedy:
                real_time_solver.original_task_graph.solve_graph_greedy()
                greedy_rewards[trial] = real_time_solver.original_task_graph.reward_model.flow_cost(real_time_solver.original_task_graph.pruned_rounded_greedy_solution)
        else:
            if solve_greedy:
                real_time_solver.original_task_graph.solve_graph_greedy()
            if perturbation == 'catastrophic':
                perturbed_reward, nodewise_perturbed_reward = real_time_solver.original_task_graph.reward_model.flow_cost_perturbed(real_time_solver.original_solution, perturbation, real_time_solver.catastrophic_perturbations)
                if solve_greedy:
                    perturbed_greedy_reward,_ = real_time_solver.original_task_graph.reward_model.flow_cost_perturbed(real_time_solver.original_task_graph.pruned_rounded_greedy_solution, perturbation, real_time_solver.catastrophic_perturbations)
            else:
                perturbed_reward,_ = real_time_solver.original_task_graph.reward_model.flow_cost_perturbed(real_time_solver.original_solution, perturbation, perturbation_params)
                if solve_greedy:
                    perturbed_greedy_reward,_ = real_time_solver.original_task_graph.reward_model.flow_cost_perturbed(real_time_solver.original_task_graph.pruned_rounded_greedy_solution, perturbation, perturbation_params)
            original_rewards[trial].append(-perturbed_reward)
            if solve_greedy:
                greedy_rewards[trial].append(-perturbed_greedy_reward)

        reward_diffs = real_time_solver.original_task_graph.reward_model._nodewise_optim_cost_function(real_time_solver.original_solution) - nodewise_perturbed_reward
        total_pert_quant = np.sum(np.abs(reward_diffs))
        for ind in real_time_solver.catastrophic_perturbations:
            reward_diffs[ind] = 0
        indirect_pert_quant = np.sum(np.abs(reward_diffs))
        indirectly_impacted_nodes = np.argwhere(reward_diffs)
        print(f"total: {total_pert_quant}, indirect: {indirect_pert_quant}, perturbed: {real_time_solver.catastrophic_perturbations}, indirectly impacted: {indirectly_impacted_nodes}")
        total_c_perturbations[trial].append(total_pert_quant)
        indirect_c_perturbations[trial].append(indirect_pert_quant)

        real_time_rewards[trial].append(np.sum(real_time_solver.task_rewards))
        all_trial_reward_perturbations_rel[trial].append(reward_perturbations_rel)
        all_trial_reward_perturbations_abs[trial].append(reward_perturbations_abs)
        print("CURRENT SOLUTION: ")
        print(real_time_solver.current_solution)
        print("ORIGINAL SOLUTION: ")
        print(real_time_solver.original_solution)
        print("CURRENT REWARDS: ")
        print(real_time_solver.task_rewards)
        print(np.sum(real_time_solver.task_rewards))
        print("ORIGINAL REWARDS: ")
        print(real_time_solver.original_rewards)
        print(np.sum(real_time_solver.original_rewards))
        print("RELATIVE REWARD PERTURBATIONS:" )
        print(reward_perturbations_rel)
        real_time_solver.print_trial_history()
    sys.stdout = sys.__stdout__
    log_file.close()
    print(f"COMPLETED TRIAL {trial}")


all_trial_perturbation_means = []
all_trial_perturbation_means_abs = []
all_trial_perturbation_means_no_err = []
for (trial, trial_abs) in zip(all_trial_reward_perturbations_rel, all_trial_reward_perturbations_abs):
    all_trial_perturbation_means_abs.append(np.mean(np.abs(np.hstack(trial_abs))))
    all_trial_perturbation_means.append(np.mean(np.abs(np.hstack(trial))))
    no_err_trial = np.nan_to_num(np.hstack(trial), posinf=0.0, neginf=0.0)
    all_trial_perturbation_means_no_err.append(np.mean(np.abs(no_err_trial)))


results_dict = {}
results_dict['original_rewards'] = original_rewards
results_dict['real_time_rewards'] = real_time_rewards
results_dict['greedy_rewards'] = greedy_rewards
#reward_advantage_all_tests = np.array(real_time_rewards) - np.array(original_rewards)
reward_advantage_all_tests = np.array(real_time_rewards) - np.transpose(np.atleast_2d(np.mean(np.array(original_rewards), axis=1)))
reward_advantage_per_trial = np.mean(reward_advantage_all_tests, axis=1)
reward_std_per_trial = np.std(original_rewards, axis=1)
reward_std_per_trial_realtime = np.std(real_time_rewards, axis=1)
try:
    results_dict['greedy_reward_average_per_trial'] = np.mean(greedy_rewards, axis=1)
except:
    import pdb; pdb.set_trace()

results_dict['reward_std_per_trial_baseline'] = reward_std_per_trial
results_dict['reward_std_per_trial_realtime'] = reward_std_per_trial_realtime
results_dict['reward_advantage'] = reward_advantage_per_trial
results_dict['reward_advantage_percent'] = results_dict['reward_advantage'] / np.mean(original_rewards,axis=1)
results_dict['reward_advantage_percent_mean'] = np.mean(results_dict['reward_advantage_percent'])
results_dict['reward_advantage_pct_no_err'] = [r for r in results_dict['reward_advantage_percent'] if r > 0]
results_dict['reward_advantage_pct_no_err_mean'] = np.mean(results_dict['reward_advantage_pct_no_err'])
results_dict['reward_advantage_pct_no_err_std'] = np.std(results_dict['reward_advantage_pct_no_err'])
results_dict['all_trial_reward_perturbations'] = all_trial_reward_perturbations_rel
results_dict['all_trial_perturbations_means'] = all_trial_perturbation_means
results_dict['all_trial_perturbations_abs'] = all_trial_perturbation_means_abs
results_dict['all_trial_reward_perturbations_no_err'] = all_trial_perturbation_means_no_err
results_dict['overall_perturbation_mean'] = np.mean(all_trial_perturbation_means)
results_dict['overall_perturbation_mean_no_err'] = np.mean(all_trial_perturbation_means_no_err)

try:
    if solve_greedy:
        results_dict['baseline_by_greedy_means'] = np.mean(original_rewards, axis=1) / np.mean(greedy_rewards, axis=1)
        results_dict['realtime_by_greedy_means'] = np.mean(real_time_rewards, axis=1) / np.mean(greedy_rewards, axis=1)
        fig, ax = plt.subplots()
        ax.plot(results_dict['baseline_by_greedy_means'], label='baseline normed by greedy')
        ax.plot(results_dict['realtime_by_greedy_means'], label='realtime normed by greedy')
        ax.legend()
        plt.savefig(str(experiment_dir / 'greedy_normed_means_plot.png'))

except:
    pass


if single_model_noise_test:
    # log reward advantages (averaged over each set of tests with the same noise level) per noise level per trial
    per_noise_level_r_adv_pct_means_normed_all_trials = []
    per_noise_level_r_adv_pct_means_all_trials = []
    per_noise_level_baseline_means_all_trials = []
    per_noise_level_it_means_all_trials = []
    per_noise_level_greedy_means_all_trials = []
    per_noise_level_total_perturbations_all_trials = []
    per_noise_level_indirect_perturbations_all_trials = []
    for trial in range(num_trials):
        per_noise_level_r = [[] for _ in range(len(noise_levels))]
        per_noise_level_baseline = [[] for _ in range(len(noise_levels))]
        per_noise_level_greedy = [[] for _ in range(len(noise_levels))]
        per_noise_level_total_perturbations = [[] for _ in range(len(noise_levels))]
        per_noise_level_indirect_perturbations = [[] for _ in range(len(noise_levels))]
        for test in range(num_tests_per_trial):
            cur_noise_level_ind = int(np.floor(test/tests_per_noise_level))
            per_noise_level_r[cur_noise_level_ind].append(real_time_rewards[trial][test])
            per_noise_level_baseline[cur_noise_level_ind].append(original_rewards[trial][test])
            per_noise_level_total_perturbations[cur_noise_level_ind].append(total_c_perturbations[trial][test])
            per_noise_level_indirect_perturbations[cur_noise_level_ind].append(indirect_c_perturbations[trial][test])
            if solve_greedy:
                per_noise_level_greedy[cur_noise_level_ind].append(greedy_rewards[trial][test])
        per_noise_level_baseline_means = [np.mean([r if r < 10000 else 0.0 for r in b_r]) for b_r in per_noise_level_baseline]
        per_noise_level_baseline_means_all_trials.append(per_noise_level_baseline_means)
        per_noise_level_it_means_all_trials.append([np.mean(it_r) for it_r in per_noise_level_r])
        per_noise_level_total_perturbations_all_trials.append(per_noise_level_total_perturbations)
        per_noise_level_indirect_perturbations_all_trials.append(per_noise_level_indirect_perturbations)
        if solve_greedy:
            per_noise_level_greedy_means_all_trials.append([np.mean(greedy_r) for greedy_r in per_noise_level_greedy])
        per_noise_level_r_adv = np.array(per_noise_level_r) - np.transpose(np.atleast_2d(per_noise_level_baseline_means))
        per_noise_level_r_adv = [[r if abs(r)<1000 else 0.0 for r in arr] for arr in per_noise_level_r_adv]
        per_noise_level_r_adv_pct = np.array(per_noise_level_r_adv) / np.transpose(np.atleast_2d(per_noise_level_baseline_means))
        per_noise_level_r_adv_pct_means = np.mean(per_noise_level_r_adv_pct, axis=1)
        per_noise_level_r_adv_pct_means_normed = per_noise_level_r_adv_pct_means / per_noise_level_r_adv_pct_means[0]
        per_noise_level_r_adv_pct_means_all_trials.append(per_noise_level_r_adv_pct_means)
        per_noise_level_r_adv_pct_means_normed_all_trials.append(per_noise_level_r_adv_pct_means_normed)
    results_dict['per_noise_level_r_adv_pct_means_all_trials'] = per_noise_level_r_adv_pct_means_all_trials
    results_dict['per_noise_level_r_adv_pct_means_normed_all_trials'] = per_noise_level_r_adv_pct_means_normed_all_trials
    results_dict['per_noise_level_baseline_means_all_trials'] = per_noise_level_baseline_means_all_trials
    results_dict['noise_vs_baseline'] = np.mean(per_noise_level_baseline_means_all_trials, axis=0)
    results_dict['noise_vs_total_perturbations'] = np.mean(np.mean(per_noise_level_total_perturbations_all_trials, axis=0), axis=1)
    results_dict['noise_vs_indirect_perturbations'] = np.mean(np.mean(per_noise_level_indirect_perturbations_all_trials, axis=0),axis=1)

    results_dict['per_noise_level_it_means_all_trials'] = per_noise_level_it_means_all_trials
    results_dict['noise_vs_it_r'] = np.mean(per_noise_level_it_means_all_trials, axis=0)
    results_dict['noise_vs_r_adv'] = np.mean(per_noise_level_r_adv_pct_means_all_trials, axis=0)
    results_dict['noise_vs_r_adv_normed'] = np.mean(per_noise_level_r_adv_pct_means_normed_all_trials, axis=0)
    # iterate through per-test reward advantages

results_dict['experiment_parameters'] = {
    'experiment_config': args.cfg,
    'perturbation_type': perturbation,
    'perturbation_param': perturbation_params,
    'perturb_model': perturb_model,
    'num_trials': num_trials,
    'num_tests_per_trial': num_tests_per_trial,
    'single_model_noise_test': single_model_noise_test,
    'tests_per_noise_level': tests_per_noise_level,
    'noise_levels': noise_levels

}
end = time.time()
results_dict['time_elapsed'] = end-start

results_file = experiment_dir / 'results.toml'
if single_model_noise_test:
    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.plot(noise_levels, np.mean(per_noise_level_baseline_means_all_trials, axis=0), label='baseline')
    ax1.plot(noise_levels, np.mean(per_noise_level_it_means_all_trials, axis=0), label='iterative')
    if solve_greedy:
        ax1.plot(noise_levels, np.mean(per_noise_level_greedy_means_all_trials, axis=0), label='greedy')
    ax1.plot(noise_levels, np.mean(per_noise_level_it_means_all_trials, axis=0) - np.mean(per_noise_level_baseline_means_all_trials, axis=0), label='advantage')
    ax1.legend()

    ax2.plot(noise_levels, np.mean(per_noise_level_r_adv_pct_means_all_trials, axis=0))
    plt.savefig(str(experiment_dir / 'means_plot.png'))
with open(str(results_file), "w") as f:
    toml.dump(results_dict, f)
