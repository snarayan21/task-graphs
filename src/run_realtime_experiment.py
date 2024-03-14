import traceback

from real_time_solver import RealTimeSolver
import pathlib
import toml
import numpy as np
import argparse
from experiment_generator import ExperimentGenerator, clean_dir_name
import sys

# create new experiment args
parser = argparse.ArgumentParser(description='Run a real-time reallocation experiment')
parser.add_argument('-cfg', default=None, help='Specify path to the experiment toml file')
parser.add_argument('-outputpath', '-o', default=None, help='Base name of output experiment dir')
parser.add_argument('-inputargs', default=None, help='Do not use, included for compatibility')
parser.add_argument('-debug', default=False, action='store_true', help='Run trials in debug mode, output to terminal' )
args = parser.parse_args()

# todo break this into sub arg dict specific to experiment generator
experiment_generator = ExperimentGenerator(args)

experiment_data_dir = pathlib.Path("experiment_data/")
experiment_data_dir.mkdir(parents=True, exist_ok=True)

all_args = toml.load(args.cfg)
exp_args = all_args['exp']

if args.outputpath is not None:
    experiment_dir_name = args.outputpath
else:
    experiment_dir_name = exp_args['exp_name']

experiment_dir_path, experiment_dir_name = clean_dir_name(experiment_dir_name, experiment_data_dir)
experiment_dir = experiment_data_dir / experiment_dir_name
experiment_dir.mkdir(parents=True, exist_ok=False)

real_time_args_file = experiment_dir / "real_time_args.toml"
out_args_dict = {}
for key, value in vars(args).items():
    out_args_dict[key] = value
out_args_dict['cfg_file_contents'] = all_args

with open(real_time_args_file, "w") as f:
    toml.dump(out_args_dict, f)

perturbation = 'gaussian'
perturbation_params = 0.1
perturb_model = False # perturb model parameters to simulate model error

num_trials = 10
original_rewards = []
real_time_rewards = []
all_trial_reward_perturbations_rel = []
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
    reward_perturbations_rel = []
    try:
        real_time_solver = RealTimeSolver(trial_args['exp'], trial_dir=str(trial_dir))
        solver_done = False
        while not solver_done:

            # TODO when integrating perturbations to reward_model parameters
            if perturb_model:
                # TODO not yet implemented
                solver_done, actual_reward, r_exp = real_time_solver.sim_step_perturbed_model(perturbation, perturbation_params)
            else:
                solver_done, actual_reward, r_exp = real_time_solver.sim_step(perturbation, perturbation_params)

            reward_perturbations_rel.append((actual_reward - r_exp) / actual_reward)

    except Exception as e:
        print(e)
        print(traceback.format_exc())
        original_rewards.append(1000000000)
        real_time_rewards.append(0)
        continue

    original_rewards.append(np.sum(real_time_solver.original_rewards))
    real_time_rewards.append(np.sum(real_time_solver.task_rewards))
    all_trial_reward_perturbations_rel.append(reward_perturbations_rel)
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
for trial in all_trial_reward_perturbations_rel:
    all_trial_perturbation_means.append(np.mean(np.abs(trial)))


results_dict = {}
results_dict['original_rewards'] = original_rewards
results_dict['real_time_rewards'] = real_time_rewards
results_dict['reward_advantage'] = np.array(real_time_rewards) - np.array(original_rewards)
results_dict['reward_advantage_percent'] = results_dict['reward_advantage'] / np.array(original_rewards)
results_dict['reward_advantage_percent_mean'] = np.mean(results_dict['reward_advantage_percent'])
results_dict['all_trial_reward_perturbations'] = all_trial_reward_perturbations_rel
results_dict['all_trial_perturbations_means'] = all_trial_perturbation_means
results_dict['overall_perturbation_mean'] = np.mean(all_trial_perturbation_means)
results_file = experiment_dir / 'results.toml'
with open(str(results_file), "w") as f:
    toml.dump(results_dict, f)