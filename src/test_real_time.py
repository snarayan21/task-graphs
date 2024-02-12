from real_time_solver import RealTimeSolver
import pathlib
import toml
import numpy as np
#exp_filepath = '/home/walker/Documents/Work/task-graphs/src/experiment_data/tasks_iterative_1-10_10_exp_0/trial_4/args.toml'
exp_filepath = '/home/walker/Documents/Work/task-graphs/src/experiment_data/real_time_test_0/trial_9/args.toml'
exp_file = pathlib.Path(exp_filepath)
all_args = toml.load(exp_filepath)

real_time_solver = RealTimeSolver(all_args['exp'])
solver_done = False
while not solver_done:
    solver_done = real_time_solver.sim_step()

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
