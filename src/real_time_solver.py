from taskgraph import TaskGraph
import numpy as np

class RealTimeSolver():

    def __init__(self):
        # initialize task graph with arguments
        # solve task graph
        # keep track of free and busy agents, completed and incomplete tasks, reward
        # model, etc
        pass

    def step(self):
        # takes in an update from graph manager node on current graph status.
        # uses update to generate a new task allocation plan

        pass

    def sim_step(self):
        # simulates a step forward in time, calls step function with updated graph
        # will be used to implement disturbances into the simulator to test disturbance
        # response/ robustness in stripped down simulator
        pass


