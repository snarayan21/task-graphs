#import cvxpy
import numpy as np
from ddp_gym.ddp_gym import DDP

# define graph
num_nodes = 4
edges = [(0,1), (1,2), (2,3)]
influence_coeffs = [1,1,1]
coalition_coeffs = [1,1,1]

def dynamics(r, f, l):
    return r*influence_coeffs[l]+f*coalition_coeffs[l]

ddp = DDP([lambda x, u: dynamics(x, u, l) for l in range(3)],  # x(i+1) = f(x(i), u)
          lambda x, u: x,  # l(x, u)
          lambda x: x,  # lf(x)
          100,
          1,
          pred_time=3)

u_seq = np.zeros((3,))
x_seq = [0]
for l in range(1,ddp.pred_time):
    x_seq.append(dynamics(x_seq[l-1],u_seq[l],l))
print(x_seq)
for i in range(3):
    k_seq, kk_seq = ddp.backward(x_seq, u_seq)
    x_seq, u_seq = ddp.forward(x_seq, u_seq, k_seq, kk_seq)