import sys
import matplotlib.pyplot as plt
from taskgraph import TaskGraph
from log_data import LogData
import toml
import argparse


def main():
    parser = argparse.ArgumentParser(description='Do a whole experiment.')
    parser.add_argument('-cfg', default=None, help='Specify path to the toml file')
    args = parser.parse_args()

    track_args = toml.load(args.cfg)

    task_planning = TaskGraph(**track_args['exp'])

    data_logger = LogData(track_args['max_steps'],
                          track_args['n_agents'],
                          track_args['num_tasks'],
                          len(track_args['edges']),
                          track_args['scenario'])

    task_planning.initializeSolver()
    task_planning.solveGraph()

    for time in range(track_args['exp']['max_steps']):

        # solve the flow optimization problem
        task_planning.solveGraph()

        # sample actual rewards (task execution)
        task_planning.simulate_task_execution()

        # update the visualization
        task_planning.render()

        # induce a disturbance or change in task characteristics
        task_planning.update_reward_curves()

        #store data
        data_logger.store_in_loop(time, task_planning.flow, task_planning.reward)

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()
