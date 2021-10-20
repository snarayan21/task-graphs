import cvxpy
import numpy as np
from ddp_gym.ddp_gym import DDP

# define graph
num_nodes = 3
edges = [(0,1), (1,2), (2,3)]
influence_coeffs = [1,1,1]
coalition_coeffs = [1,1,1]

def dynamics(r, f, l):
    return r*influence_coeffs[l]+f*coalition_coeffs[l]

ddp = DDP([lambda x, u: dynamics(x, u, l) for l in range(3)],  # x(i+1) = f(x(i), u)
          lambda x, u: x,  # l(x, u)
          lambda x: x,  # lf(x)
          100,
          1)

# main DDP iteration
converged = False
while not converged:
    # backward pass
    for l in range(num_nodes-1,0,-1):
        calc_perturbation_func(l)


    # forward pass
    for i in range(num_nodes):
        print(i)
        if i == num_nodes-1:
            converged = True
