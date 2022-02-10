import sys
import matplotlib.pyplot as plt
from taskgraph import TaskGraph
from log_data import LogData
import toml
import argparse

# LP main program
def main():
    parser = argparse.ArgumentParser(description='Do a whole experiment.')
    parser.add_argument('-cfg', default=None, help='Specify path to the toml file')
    parser.add_argument('-milp', action='store_true', default=False, help='include -milp flag to additionally solve graph with MILP')
    args = parser.parse_args()

    track_args = toml.load(args.cfg)

    task_planning = TaskGraph(**track_args['exp'])

    task_planning.initialize_solver_ddp(**track_args['ddp'])
    task_planning.solve_ddp()

    if args.milp:
        milp_solver = TaskGraph(**track_args['exp'])
        milp_solver.initializeSolver()
        milp_solver.solveGraph()

    #breakpoint()
    """ for time in range(10):
        # induce a disturbance or change in task characteristics
        # task_planning.update_reward_curves()  # TODO: introduce adaptive piece here

        # resolve the problem with the modified scenario
        task_planning.solve_ddp()

        # sample actual rewards (task execution)
        task_planning.simulate_task_execution()

        # update the visualization
        task_planning.render()

        # induce a disturbance or change in task characteristics
        task_planning.update_reward_curves()

        #store data
        #data_logger.store_in_loop(time, task_planning.flow, task_planning.reward) """

    #data_logger.write_to_file()
    #task_planning.render()
    #plt.ioff()
    #plt.show()


if __name__ == '__main__':
    main()
