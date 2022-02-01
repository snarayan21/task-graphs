#import cvxpy
import numpy as np
from ddp_gym.ddp_gym import DDP
import networkx as nx

# define graph
num_nodes = 3
edges = [[0,1], [1,2]]
influence_coeffs = [None,1,1]
coalition_coeffs = [None,1,1]
#use networkx to generate incidence matrix for us
task_graph = nx.DiGraph()
task_graph.add_nodes_from(range(num_nodes))
task_graph.add_edges_from(edges)
incidence_mat = np.array(nx.linalg.graphmatrix.incidence_matrix(task_graph, oriented=True).A)
print(incidence_mat)

def dynamics(r, f, l):
    return r*r*influence_coeffs[l]+f*f*coalition_coeffs[l]

def dynamics_b(r,f,l):
    return -(r**2) + influence_coeffs[l]*r - (f**2) + coalition_coeffs[l]*f

def dynamics_c(r,f,l):
    return (r**2) + 2*influence_coeffs[l]*r + coalition_coeffs[l]*f

dynamics_f = dynamics_b

ddp = DDP([lambda x, u: dynamics_f(x, u, l) for l in range(num_nodes-1)],  # x(i+1) = f(x(i), u)
          lambda x, u: -x,  # l(x, u)
          lambda x: -x,  # lf(x)
          1000,
          1,
          num_nodes-1,
          incidence_mat,
          constraint_type='None')

u_seq = np.ones((num_nodes-1,))
x_seq = [0.01]
for l in range(0,ddp.pred_time):
    x_seq.append(dynamics_f(x_seq[l],u_seq[l],l+1))
print(x_seq)

prev_u_seq = np.zeros_like(np.array(u_seq))
i = 0
max_iter = 30
threshold = -1
delta = np.inf
while i < max_iter and delta > threshold:
    k_seq, kk_seq = ddp.backward(x_seq, u_seq)
    x_seq, u_seq = ddp.forward(x_seq, u_seq, k_seq, kk_seq)
    print('x_seq: ',x_seq, ' total_reward: ', np.sum(x_seq))
    print('u_seq: ',u_seq)
    i += 1
    delta = np.linalg.norm(np.array(u_seq) - np.array(prev_u_seq))
    print('delta: ',delta)
    prev_u_seq = u_seq
