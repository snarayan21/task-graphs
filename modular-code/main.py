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

    print(task_planning.num_tasks)
    print(task_planning.edges)

    task_planning.initializeSolver()

    for time in range(100000):

        # induce a disturbance or change in task characteristics
        task_planning.update_reward_curves()

        # resolve the problem with the modified scenario
        task_planning.solveGraph()

        # update the visualization
        task_planning.render()

    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()

    