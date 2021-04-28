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

    graph = TaskGraph(**track_args['exp'])

    print(graph.num_tasks)
    print(graph.edges)

    graph.initializeSolver()

    graph.solveGraph()


if __name__ == '__main__':
    main()

    