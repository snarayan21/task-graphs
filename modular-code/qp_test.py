#import cvxpy
import numpy as np
from ddp_gym.ddp_gym import DDP
import networkx as nx

# define graph
num_nodes = 4
edges = [[0,1], [1,2], [2,3]]
influence_coeffs = [None,1,1,1]
coalition_coeffs = [None,1,1,1]
#use networkx to generate incidence matrix for us
task_graph = nx.DiGraph()
task_graph.add_nodes_from(range(num_nodes))
task_graph.add_edges_from(edges)
incidence_mat = np.array(nx.linalg.graphmatrix.incidence_matrix(task_graph, oriented=True).A)
print(incidence_mat)

def dynamics(r, f, l):
    return r*r*influence_coeffs[l]+f*coalition_coeffs[l]

ddp = DDP([lambda x, u: dynamics(x, u, l) for l in range(3)],  # x(i+1) = f(x(i), u)
          lambda x, u: x,  # l(x, u)
          lambda x: x,  # lf(x)
          1,
          1,
          3,
          incidence_mat)

u_seq = np.zeros((3,))
x_seq = [-1.0]
for l in range(0,ddp.pred_time):
    x_seq.append(dynamics(x_seq[l],u_seq[l],l+1))
print(x_seq)
for i in range(300):
    k_seq, kk_seq = ddp.backward(x_seq, u_seq)
    x_seq, u_seq = ddp.forward(x_seq, u_seq, k_seq, kk_seq)
    print(x_seq)
    print(u_seq)
