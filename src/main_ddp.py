import sys
import matplotlib.pyplot as plt
from taskgraph import TaskGraph
from log_data import LogData
import toml
import argparse
from draw_construction import graph_tower, flows_to_taskrobots

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
    #task_graph.solve_graph_minlp()

    s,f = task_graph.time_task_execution(task_graph.last_baseline_solution.x)
    heights = []
    blocks = []

    print("-----------------BASELINE SOLUTION TIMES:-----------------")
    print('task start times: ', s)
    print('task finish times: ', f)
    print('task durations: ', task_graph.task_times)

    #HOW TO GENERATE AND RUN TOWER
    #1. pull from autonomous_construction branch
    #2. make sure you are in task-graphs/src
    #3. run: python ./autonomous_construction/initial_blocks.py [tower base width] [# of tower layers] [tower name]
    #4. run: python main_ddp.py -cfg ./autonomous_construction/generated_examples/[tower name].toml [whatver flags / args]
    #Note the use of the flows_to_taskrobots and graph_tower functions below. Imported from src/draw_construction.py
    
    if("tower" in track_args):
        heights = track_args["tower"]["heights"]
        blocks = track_args["tower"]["blocks"]
        totrobots = 20
        taskrobots = flows_to_taskrobots(task_graph.last_baseline_solution.x, track_args['exp']["edges"], track_args['exp']["num_tasks"], totrobots)
        graph_tower(s, f, totrobots, taskrobots, heights, blocks, track_args['exp']['coalition_params'], (args.cfg).split("/")[-1].split(".")[0])

    print('DDP solution: ')
    print(task_graph.last_ddp_solution)
    ddp_reward = task_graph.reward_model.flow_cost(task_graph.last_ddp_solution)

    print('DDP solution reward: ', ddp_reward)

    print('MINLP solution reward: ')
    print(task_graph.last_minlp_solution_val)

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
    plt.plot(task_graph.constraint_violation, 'r')
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
