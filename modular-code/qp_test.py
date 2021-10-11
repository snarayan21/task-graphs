import cvxpy
import numpy as np

def calc_perturbation_func(i):
    # set up linear program


# define graph
num_nodes = 3
edges = [(0,1), (1,2), (2,3)]
influence_coeffs = [1,1,1]
coalition_coeffs = [1,1,1]



# main DDP iteration
converged = False
while not converged:
    # backwards pass
    for i in range(num_nodes-1,0,-1):
        calc_perturbation_func(i)


    # forwards pass
    for i in range(num_nodes):
        print(i)
        if i == num_nodes-1:
            converged = True