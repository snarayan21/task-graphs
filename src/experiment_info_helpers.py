import json
import numpy as np
import pathlib


class ExperimentData:

    def __init__(self, jsonfilepath):
        self.exp_data = json.load(pathlib.Path(jsonfilepath).absolute())

    def get_args_exp_trial(self, exp, trialnum):
        return self.exp_data[exp]["trial" + str(trialnum)]["args"]

    def get_results_exp_trial(self, exp, trialnum):
        return self.exp_data[exp]["trial" + str(trialnum)]["results"]

    def get_exp_avg_tasks(self, exp):
        tasks = []
        for trial in self.exp_data[exp]:
            tasks.append(self.exp_data[exp][trial]["args"]["exp"]["num_tasks"])

        tasks = np.array(tasks)
        return np.mean(tasks), np.std(tasks)

    def get_exp_avg_edges(self, exp):
        edges = []
        for trial in self.exp_data[exp]:
            edges.append(len(self.exp_data[exp][trial]["args"]["exp"]["edges"]))

        edges = np.array(edges)
        return np.mean(edges), np.std(edges)

    def get_exp_avg_robots(self, exp):
        robots = []
        for trial in self.exp_data[exp]:
            robots.append(self.exp_data[exp][trial]["args"]["exp"]["num_robots"])

        robots = np.array(robots)
        return np.mean(robots), np.std(robots)

    def get_exp_avg_makespan_constraint(self, exp):
        makespan_constraint = []
        for trial in self.exp_data[exp]:
            makespan_constraint.append(self.exp_data[exp][trial]["args"]["exp"]["makespan_constraint"])

        makespan_constraint = np.array(makespan_constraint)
        return np.mean(makespan_constraint), np.std(makespan_constraint)

    def get_exp_trial_task_times(self, exp, trialnum):
        return self.exp_data[exp]["trial" + str(trialnum)]["args"]["exp"]["task_times"]

    def get_exp_trial_baseline_reward(self, exp, trialnum):
        return self.exp_data[exp]["trial" + str(trialnum)]["results"]["baseline_reward"]

    def get_exp_trial_baseline_task_rewards(self, exp, trialnum):
        return self.exp_data[exp]["trial" + str(trialnum)]["results"]["baseline_task_rewards"]

    def get_exp_trial_baseline_solution_time(self, exp, trialnum):
        return self.exp_data[exp]["trial" + str(trialnum)]["results"]["baseline_solution_time"]

    def get_exp_trial_baseline_makespan(self, exp, trialnum):
        return self.exp_data[exp]["trial" + str(trialnum)]["results"]["baseline_makespan"]

    def get_exp_trial_pruned_baseline_solution(self, exp, trialnum):
        return self.exp_data[exp]["trial" + str(trialnum)]["results"]["pruned_baseline_solution"]

    def get_exp_trial_pruned_baseline_reward(self, exp, trialnum):
        return self.exp_data[exp]["trial" + str(trialnum)]["results"]["pruned_baseline_reward"]

    def get_exp_trial_pruned_baseline_task_rewards(self, exp, trialnum):
        return self.exp_data[exp]["trial" + str(trialnum)]["results"]["pruned_baseline_task_rewards"]

    def get_exp_trial_pruned_baseline_solution_time(self, exp, trialnum):
        return self.exp_data[exp]["trial" + str(trialnum)]["results"]["pruned_baseline_solution_time"]

    def get_exp_trial_pruned_baseline_makespan(self, exp, trialnum):
        return self.exp_data[exp]["trial" + str(trialnum)]["results"]["pruned_baseline_makespan"]

    def get_exp_trial_rounded_baseline_solution(self, exp, trialnum):
        return self.exp_data[exp]["trial" + str(trialnum)]["results"]["rounded_baseline_solution"]

    def get_exp_trial_rounded_baseline_reward(self, exp, trialnum):
        return self.exp_data[exp]["trial" + str(trialnum)]["results"]["rounded_baseline_reward"]

    def get_exp_trial_greedy_solution(self, exp, trialnum):
        return self.exp_data[exp]["trial" + str(trialnum)]["results"]["greedy_solution"]

    def get_exp_trial_greedy_reward(self, exp, trialnum):
        return self.exp_data[exp]["trial" + str(trialnum)]["results"]["greedy_reward"]

    def get_exp_trial_greedy_execution_times(self, exp, trialnum):
        return self.exp_data[exp]["trial" + str(trialnum)]["results"]["greedy_execution_times"]

    def get_exp_trial_greedy_solution_time(self, exp, trialnum):
        return self.exp_data[exp]["trial" + str(trialnum)]["results"]["greedy_solution_time"]

    def get_exp_trial_greedy_makespan(self, exp, trialnum):
        return self.exp_data[exp]["trial" + str(trialnum)]["results"]["greedy_makespan"]

    def get_exp_trial_pruned_greedy_solution(self, exp, trialnum):
        return self.exp_data[exp]["trial" + str(trialnum)]["results"]["pruned_greedy_solution"]

    def get_exp_trial_pruned_greedy_reward(self, exp, trialnum):
        return self.exp_data[exp]["trial" + str(trialnum)]["results"]["pruned_greedy_reward"]

    def get_exp_trial_pruned_greedy_makespan(self, exp, trialnum):
        return self.exp_data[exp]["trial" + str(trialnum)]["results"]["pruned_greedy_makespan"]

    def get_exp_trial_pruned_rounded_greedy_solution(self, exp, trialnum):
        return self.exp_data[exp]["trial" + str(trialnum)]["results"]["pruned_rounded_greedy_solution"]

    def get_exp_trial_pruned_rounded_greedy_reward(self, exp, trialnum):
        return self.exp_data[exp]["trial" + str(trialnum)]["results"]["pruned_rounded_greedy_reward"]

    def get_exp_trial_minlp_solution(self, exp, trialnum):
        return self.exp_data[exp]["trial" + str(trialnum)]["results"]["minlp_solution"]

    def get_exp_trial_minlp_reward(self, exp, trialnum):
        return self.exp_data[exp]["trial" + str(trialnum)]["results"]["minlp_reward"]

    def get_exp_trial_minlp_execution_times(self, exp, trialnum):
        return self.exp_data[exp]["trial" + str(trialnum)]["results"]["minlp_execution_times"]

    def get_exp_trial_minlp_solution_time(self, exp, trialnum):
        return self.exp_data[exp]["trial" + str(trialnum)]["results"]["minlp_solution_time"]

    def get_exp_trial_minlp_makespan(self, exp, trialnum):
        return self.exp_data[exp]["trial" + str(trialnum)]["results"]["minlp_makespan"]

    def get_exp_avg_baseline_reward(self, exp):
        baseline_reward = []
        for trial in self.exp_data[exp]:
            baseline_reward.append(self.exp_data[exp][trial]["results"]["baseline_reward"])

        baseline_reward = np.array(baseline_reward)
        return np.mean(baseline_reward), np.std(baseline_reward)

    def get_exp_avg_baseline_solution_time(self, exp):
        baseline_solution_time = []
        for trial in self.exp_data[exp]:
            baseline_solution_time.append(self.exp_data[exp][trial]["results"]["baseline_solution_time"])

        baseline_solution_time = np.array(baseline_solution_time)
        return np.mean(baseline_solution_time), np.std(baseline_solution_time)

    def get_exp_avg_baseline_makespan(self, exp):
        baseline_makespan = []
        for trial in self.exp_data[exp]:
            baseline_makespan.append(self.exp_data[exp][trial]["results"]["baseline_makespan"])

        baseline_makespan = np.array(baseline_makespan)
        return np.mean(baseline_makespan), np.std(baseline_makespan)

    def get_exp_avg_pruned_baseline_reward(self, exp):
        pruned_baseline_reward = []
        num_tasks = 0.0
        for trial in self.exp_data[exp]:
            pruned_baseline_reward.append(self.exp_data[exp][trial]["results"]["pruned_baseline_reward"])

        pruned_baseline_reward = np.array(pruned_baseline_reward)
        return np.mean(pruned_baseline_reward), np.std(pruned_baseline_reward)

    def get_exp_avg_pruned_baseline_makespan(self, exp):
        pruned_baseline_makespan = []
        for trial in self.exp_data[exp]:
            pruned_baseline_makespan.append(self.exp_data[exp][trial]["results"]["pruned_baseline_makespan"])

        pruned_baseline_makespan = np.array(pruned_baseline_makespan)
        return np.mean(pruned_baseline_makespan), np.std(pruned_baseline_makespan)

    def get_exp_avg_pruned_rounded_baseline_reward(self, exp):
        pruned_rounded_baseline_reward = []
        for trial in self.exp_data[exp]:
            pruned_rounded_baseline_reward.append(self.exp_data[exp][trial]["results"]["pruned_rounded_baseline_reward"])

        pruned_rounded_baseline_reward = np.array(pruned_rounded_baseline_reward)
        return np.mean(pruned_rounded_baseline_reward), np.std(pruned_rounded_baseline_reward)

    def get_exp_avg_rounded_baseline_reward(self, exp):
        rounded_baseline_reward = []
        for trial in self.exp_data[exp]:
            rounded_baseline_reward.append(self.exp_data[exp][trial]["results"]["rounded_baseline_reward"])

        rounded_baseline_reward = np.array(rounded_baseline_reward)
        return np.mean(rounded_baseline_reward), np.std(rounded_baseline_reward)

    def get_exp_avg_greedy_reward(self, exp):
        greedy_reward = []
        for trial in self.exp_data[exp]:
            greedy_reward.append(self.exp_data[exp][trial]["results"]["greedy_reward"])

        greedy_reward = np.array(greedy_reward)
        return np.mean(greedy_reward), np.std(greedy_reward)

    def get_exp_avg_greedy_solution_time(self, exp):
        greedy_solution_time = []
        for trial in self.exp_data[exp]:
            greedy_solution_time.append(self.exp_data[exp][trial]["results"]["greedy_solution_time"])

        greedy_solution_time = np.array(greedy_solution_time)
        return np.mean(greedy_solution_time), np.std(greedy_solution_time)

    def get_exp_avg_greedy_makespan(self, exp):
        greedy_makespan = []
        for trial in self.exp_data[exp]:
            greedy_makespan.append(self.exp_data[exp][trial]["results"]["greedy_makespan"])

        greedy_makespan = np.array(greedy_makespan)
        return np.mean(greedy_makespan), np.std(greedy_makespan)

    def get_exp_avg_pruned_greedy_reward(self, exp):
        pruned_greedy_reward = []
        for trial in self.exp_data[exp]:
            pruned_greedy_reward.append(self.exp_data[exp][trial]["results"]["pruned_greedy_reward"])

        pruned_greedy_reward = np.array(pruned_greedy_reward)
        return np.mean(pruned_greedy_reward), np.std(pruned_greedy_reward)

    def get_exp_avg_pruned_greedy_makespan(self, exp):
        pruned_greedy_makespan = []
        for trial in self.exp_data[exp]:
            pruned_greedy_makespan.append(self.exp_data[exp][trial]["results"]["pruned_greedy_makespan"])

        pruned_greedy_makespan = np.array(pruned_greedy_makespan)
        return np.mean(pruned_greedy_makespan), np.std(pruned_greedy_makespan)

    def get_exp_avg_pruned_rounded_greedy_reward(self, exp):
        pruned_rounded_greedy_reward = []
        for trial in self.exp_data[exp]:
            pruned_rounded_greedy_reward.append(self.exp_data[exp][trial]["results"]["pruned_rounded_greedy_reward"])

        pruned_rounded_greedy_reward = np.array(pruned_rounded_greedy_reward)
        return np.mean(pruned_rounded_greedy_reward), np.std(pruned_rounded_greedy_reward)

    def get_exp_avg_minlp_reward(self, exp):
        minlp_reward = []
        for trial in self.exp_data[exp]:
            minlp_reward.append(self.exp_data[exp][trial]["results"]["minlp_reward"])

        minlp_reward = np.array(minlp_reward)
        return np.mean(minlp_reward), np.std(minlp_reward)

    def get_exp_avg_minlp_solution_time(self, exp):
        minlp_solution_time = []
        for trial in self.exp_data[exp]:
            minlp_solution_time.append(self.exp_data[exp][trial]["results"]["minlp_solution_time"])

        minlp_solution_time = np.array(minlp_solution_time)
        return np.mean(minlp_solution_time), np.std(minlp_solution_time)

    def get_exp_avg_minlp_makespan(self, exp):
        minlp_makespan = []
        for trial in self.exp_data[exp]:
            minlp_makespan.append(self.exp_data[exp][trial]["results"]["minlp_makespan"])

        minlp_makespan = np.array(minlp_makespan)
        return np.mean(minlp_makespan), np.std(minlp_makespan)







