#import cvxpy
import numpy as np
from ddp_gym.ddp_gym import DDP

# define graph
num_nodes = 4
edges = [(0,1), (1,2), (2,3)]
influence_coeffs = [None,2,2,2]
coalition_coeffs = [None,2,2,2]

def dynamics(r, f, l):
    return r*r*influence_coeffs[l]+f*coalition_coeffs[l]

def dynamics_b(r,f,l):
    return -(r**2) + influence_coeffs[l]*r  - (f**2) + coalition_coeffs[l]*f

ddp = DDP([lambda x, u: dynamics_b(x, u, l) for l in range(3)],  # x(i+1) = f(x(i), u)
          lambda x, u: x,  # l(x, u)
          lambda x: x,  # lf(x)
          100,
          1,
          pred_time=3)

u_seq = np.zeros((3,))
x_seq = [0.01]
for l in range(0,ddp.pred_time):
    x_seq.append(dynamics(x_seq[l],u_seq[l],l+1))
print(x_seq)

prev_u_seq = np.zeros_like(np.array(u_seq))
i = 0
max_iter = 30
threshold = 0
delta = np.inf
while i < max_iter and delta < threshold:
    k_seq, kk_seq = ddp.backward(x_seq, u_seq)
    x_seq, u_seq = ddp.forward(x_seq, u_seq, k_seq, kk_seq)
    print(x_seq)
    print(u_seq)
    i += 1
    delta = np.linalg.norm(np.array(u_seq) - np.array(prev_u_seq))
    print(delta)
    prev_u_seq = u_seq
