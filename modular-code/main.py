import sys
import matplotlib.pyplot as plt
from taskgraph import TaskGraph
import toml
import argparse


def main():
    parser = argparse.ArgumentParser(description='Do a whole experiment.')
    parser.add_argument('-cfg', default=None, help='Specify path to the toml file')
    args = parser.parse_args()

    track_args = toml.load(args.cfg)

    task_planning = TaskGraph(**track_args['exp'])

    task_planning.initializeSolver()
    task_planning.solveGraph()

    for time in range(10):
        # induce a disturbance or change in task characteristics
        # task_planning.update_reward_curves()  # TODO: introduce adaptive piece here

        # resolve the problem with the modified scenario
        task_planning.solveGraph()

        # sample actual rewards (task execution)
        task_planning.simulate_task_execution()

        # update the visualization
        task_planning.render()
        task_planning.update_reward_curves()

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()
