from real_time_solver import RealTimeSolver
import pathlib
import toml
exp_filepath = '/home/walker/Documents/Work/task-graphs/src/experiment_data/tasks_iterative_1-10_10_exp_0/trial_4/args.toml'
exp_file = pathlib.Path(exp_filepath)
all_args = toml.load(exp_filepath)
all_args['exp'].pop('max_steps')

real_time_solver = RealTimeSolver(all_args['exp'])
solver_done = False
while not solver_done:
    solver_done = real_time_solver.sim_step()
