import sys
import matplotlib.pyplot as plt
from taskgraph import TaskGraph
from log_data import LogData
import toml
import argparse

# NP main program
def main():
    parser = argparse.ArgumentParser(description='Do a single trial.')
    parser.add_argument('-cfg', default=None, help='Specify path to the toml file')
    parser.add_argument('-baseline', '-b', action='store_true', default=False, help='include -baseline (-b) flag to additionally solve graph with baseline optimizer')
    parser.add_argument('-greedy', '-g', action='store_true', default=False, help='include -greedy (-g) flag to additionally solve graph with greedy algorithm')
    args = parser.parse_args()

    track_args = toml.load(args.cfg)
    task_graph = TaskGraph(**track_args['exp'])
    if args.baseline:
        #task_graph.initializeSolver()
        #task_graph.solveGraph()
        task_graph.solve_graph_scipy()

    if args.greedy:
        task_graph.solve_graph_greedy()

    task_graph.initialize_solver_ddp(**track_args['ddp'])
    task_graph.solve_ddp()

    print('DDP solution: ')
    print(task_graph.last_ddp_solution)
    ddp_reward = task_graph.reward_model.flow_cost(task_graph.last_ddp_solution)
    optimal_reward = task_graph.reward_model.flow_cost([0.5, 0.25, 0.25, 0.25, 0.25])
    print('DDP solution reward: ', ddp_reward)
    print('Optimal solution reward: ', optimal_reward)

    if args.greedy:
        print('Greedy solution:' )
        print(task_graph.last_greedy_solution)
        g_reward = task_graph.reward_model.flow_cost(task_graph.last_greedy_solution)
        print('Greedy solution reward: ', g_reward)

    if args.baseline:
        print('Baseline solution: ')
        print(task_graph.last_baseline_solution.x)
        bl_reward = task_graph.reward_model.flow_cost(task_graph.last_baseline_solution.x)
        print('Baseline solution reward: ', bl_reward)

        print("Optimality ratio: ", ddp_reward/bl_reward)

    plt.plot(task_graph.ddp_reward_history)
    plt.xlabel('Iteration #')
    plt.ylabel('Reward')
    plt.show()


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
