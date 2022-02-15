import sys
import matplotlib.pyplot as plt
from taskgraph import TaskGraph
from log_data import LogData
import toml
import argparse

# NP main program
def main():
    parser = argparse.ArgumentParser(description='Do a whole experiment.')
    parser.add_argument('-cfg', default=None, help='Specify path to the toml file')
    parser.add_argument('-baseline', '-b', action='store_true', default=False, help='include -baseline flag to additionally solve graph with baseline optimizer')
    args = parser.parse_args()

    track_args = toml.load(args.cfg)

    if args.baseline:
        np_solver = TaskGraph(**track_args['exp'])
        #np_solver.initializeSolver()
        #np_solver.solveGraph()
        np_solver.solve_graph_scipy()

    task_planning = TaskGraph(**track_args['exp'])

    task_planning.initialize_solver_ddp(**track_args['ddp'])
    task_planning.solve_ddp()


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
