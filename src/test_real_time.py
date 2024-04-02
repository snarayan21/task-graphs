from real_time_solver import RealTimeSolver
import pathlib
import toml
import numpy as np
#exp_filepath = '/home/walker/Documents/Work/task-graphs/src/experiment_data/tasks_iterative_1-10_10_exp_0/trial_4/args.toml'
#exp_filepath = '/home/walker/Documents/Work/task-graphs/src/experiment_data/real_time_test_0/trial_9/args.toml'
exp_filepath = '/home/walker/Documents/Work/task-graphs/src/experiment_data/real_time_15_exp_109/trial_61/args.toml'
exp_file = pathlib.Path(exp_filepath)
all_args = toml.load(exp_filepath)

run_minlp = True
if run_minlp:
    all_args['exp']['run_minlp'] = True
    all_args['exp']['minlp_time_constraint'] = 6000

perturbation = 'gaussian'
perturbation_params = 0.1
perturb_model = False # perturb model parameters to simulate model error


real_time_solver = RealTimeSolver(all_args['exp'])
solver_done = False
while not solver_done:
    if perturb_model:
        # TODO not yet implemented
        solver_done, actual_reward, r_exp = real_time_solver.sim_step_perturbed_model(perturbation, perturbation_params)
    else:
        solver_done, actual_reward, r_exp = real_time_solver.sim_step(perturbation, perturbation_params)

#real_time_solver.original_task_graph.solve_graph_minlp()
r, nodewise_r = real_time_solver.original_task_graph.reward_model.flow_cost_perturbed(real_time_solver.original_solution, perturbation_params)
print("PERTURBED ORIGINAL REWARDS")
print(-r)
print(-nodewise_r)
print(f"diffs: {np.array(real_time_solver.original_rewards + nodewise_r)}")
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

real_time_solver.print_trial_history()
