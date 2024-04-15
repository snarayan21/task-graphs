import copy

import numpy as np
import matplotlib.pyplot as plt
import toml
import pathlib
# results_files = ["experiment_data/all_oracles_4-2_exp_0/results.toml",
#                  "experiment_data/all_oracles_4-2_exp_1/results.toml"]
n_tasks=10
#results_files = [f"experiment_data/all_oracles_4-6_{n_tasks}_exp_{i}/results.toml" for i in range(20)]
#results_files = [f"experiment_data/cat_4-14_exp_{i}/results.toml" for i in range(12,18)]
results_files = [f"experiment_data/cat_4-14_ntasks_15_exp_{i}/results.toml" for i in range(0,5)]
#results_files = ["experiment_data/test_zero_err_22/results.toml"]
results_tomls = []
for f in results_files:
    if pathlib.Path(f).exists():
        results_dict = toml.load(f)
        results_tomls.append(results_dict)

noise_levels = results_tomls[0]['experiment_parameters']['noise_levels']
tests_per_noise_level = results_tomls[0]['experiment_parameters']['tests_per_noise_level']
num_trials = results_tomls[0]['experiment_parameters']['num_trials']


#import pdb; pdb.set_trace()
get_std_keys = ['original_rewards', 'real_time_rewards', 'offline_oracle_rewards', 'online_oracle_rewards',
                'offline_unperturbed_rewards', 'online_unperturbed_rewards']
label_dict = {'original_rewards': 'offline perturbed',
              'real_time_rewards': 'online perturbed',
              'offline_oracle_rewards': 'offline oracle',
              'online_oracle_rewards': 'online oracle',
              'offline_unperturbed_rewards': 'offline unperturbed',
              'online_unperturbed_rewards': 'online unperturbed'}
std_data_dict = {}

stds = {}
norm_key = 'online_unperturbed_rewards'
normed_data_dict = {}
for key in get_std_keys:
    std_data_dict[key] = None
    normed_data_dict[key] = None
    for toml in results_tomls:
        data = toml[key]
        data = np.array(data, dtype=float).reshape((num_trials,len(noise_levels),tests_per_noise_level))
        data_means = np.mean(data, axis=2)
        if std_data_dict[key] is None:
            std_data_dict[key] = data_means
            normed_data_dict[key] = data
        else:
            std_data_dict[key] = np.concatenate((std_data_dict[key], data_means), axis=0)
            normed_data_dict[key] = np.concatenate((normed_data_dict[key], data), axis=0)

    stds[key] = np.std(std_data_dict[key], axis=0)
    print(len(std_data_dict[key]))

for key in get_std_keys:
    unnormed_data = copy.deepcopy(normed_data_dict[key])
    normed_data = np.divide(unnormed_data, normed_data_dict[norm_key])
    normed_data_dict[key] = normed_data


concat_original_rewards = [np.array(t['original_rewards'],dtype=float).reshape((num_trials,len(noise_levels), tests_per_noise_level)) for t in results_tomls]
concat_real_time_rewards = [np.array(t['real_time_rewards'],dtype=float).reshape((num_trials,len(noise_levels), tests_per_noise_level)) for t in results_tomls]
np_concat_original_rewards = None
np_concat_real_time_rewards = None
for i in range(len(concat_original_rewards)):
    if np_concat_original_rewards is None:
        np_concat_original_rewards = concat_original_rewards[i]
        np_concat_real_time_rewards = concat_real_time_rewards[i]
    else:
        np_concat_original_rewards = np.concatenate((np_concat_original_rewards, concat_original_rewards[i]))
        np_concat_real_time_rewards = np.concatenate((np_concat_real_time_rewards, concat_real_time_rewards[i]))

per_noise_level_original_means_per_trial = np.mean(np_concat_original_rewards, axis=2)
per_noise_level_real_time_means_per_trial = np.mean(np_concat_real_time_rewards, axis=2)
concat_noise_vs_it_r = [np.array(t['noise_vs_it_r'],dtype=float) for t in results_tomls]
concat_noise_vs_baseline = [np.array(t['noise_vs_baseline'],dtype=float) for t in results_tomls]
concat_per_noise_level_online_oracle_means = [np.mean(np.array(t['per_noise_level_online_oracle_means_all_trials'],dtype=float), axis=0) for t in results_tomls]
concat_per_noise_level_offline_oracle_means = [np.mean(np.array(t['per_noise_level_offline_oracle_means_all_trials'],dtype=float), axis=0) for t in results_tomls]
concat_per_noise_level_offline_unperturbed = [np.mean(np.array(t['per_noise_level_offline_unperturbed_means_all_trials'],dtype=float), axis=0) for t in results_tomls]
concat_per_noise_level_online_unperturbed = [np.mean(np.array(t['per_noise_level_online_unperturbed_means_all_trials'],dtype=float), axis=0) for t in results_tomls]
noise_vs_it_r = np.mean(concat_noise_vs_it_r, axis=0)
noise_vs_baseline = np.mean(concat_noise_vs_baseline, axis=0)
per_noise_level_online_oracle_means_all_trials = np.mean(concat_per_noise_level_online_oracle_means, axis=0)
per_noise_level_offline_unperturbed_all_trials = np.mean(concat_per_noise_level_offline_unperturbed, axis=0)
per_noise_level_offline_oracle_means_all_trials = np.mean(concat_per_noise_level_offline_oracle_means, axis=0)
per_noise_level_online_unperturbed_all_trials = np.mean(concat_per_noise_level_online_unperturbed, axis=0)

data_list = [noise_vs_baseline, noise_vs_it_r, per_noise_level_offline_oracle_means_all_trials, per_noise_level_online_oracle_means_all_trials,
             per_noise_level_offline_unperturbed_all_trials, per_noise_level_online_unperturbed_all_trials]

fig, (ax1, ax2) = plt.subplots(2,1)
#fig, ax1 = plt.subplots(1,1)
ax1.plot(noise_levels, noise_vs_baseline, label="offline perturbed")
ax1.plot(noise_levels, noise_vs_it_r, label="online perturbed")
ax1.plot(noise_levels, per_noise_level_offline_oracle_means_all_trials, label="offline oracle")
ax1.plot(noise_levels, per_noise_level_online_oracle_means_all_trials, label="online oracle")
ax1.plot(noise_levels, per_noise_level_offline_unperturbed_all_trials, label="offline unperturbed")
ax1.plot(noise_levels, per_noise_level_online_unperturbed_all_trials, label="online unperturbed")

for i in range(len(data_list)):
    ax1.fill_between(noise_levels, data_list[i]-stds[get_std_keys[i]], data_list[i]+stds[get_std_keys[i]], alpha=0.2)
ax1.legend()
ax1.set_xlabel("Likelihood of Catastrophic Failure")
ax1.set_ylabel("Reward")
ax1.set_title(f"{n_tasks} Task Experiment: Noise vs. Reward")

plot_normed = True
if plot_normed:
    for key in get_std_keys:
        mean_data = np.mean(np.mean(normed_data_dict[key], axis=2), axis=0)
        std_data = np.std(np.mean(normed_data_dict[key], axis=2), axis=0)

        ax2.plot(noise_levels, mean_data, label=label_dict[key])
        ax2.fill_between(noise_levels, mean_data - std_data, mean_data + std_data, alpha=0.2)

    ax2.legend()
    ax2.set_title('Reward Normalized by Online Unperturbed Rewards for Each Trial')
plt.show()
