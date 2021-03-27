from gurobipy import Model, GRB, quicksum, abs_, and_
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


#edges as adjacency list
#edges_n is normal directed graph edges
edges = {1: [2,3], 2: [4], 3: [4], 4: [5], 5: []}
#edges_tot is all graph edges connecting to some node
edges_tot = {1: [2,3], 2: [1,4], 3: [1,4], 4: [2,3,5], 5: [4]}
#ce is the c values for each edge
ce = {(2,4): 1, (3,4): 1, (4,5): 0.4}

#sets needed
#V is the vertices
V = [i for i in range(1,6)]
#Vi is intermediate vertices
Vi = [i for i in range(2,5)]
#En is the normal directed graph edges
En = []
#En is the reverse directed graph edges
Er = []

Ereg = []
for i in range(1,6):
    conn = edges[i]
    for j in conn:
        En.append((i,j))
        Er.append((j,i))
        if(not((i == 1 and j == 2) or (i == 1 and j == 3))):
            Ereg.append((i,j))

#Et is both normal and reverse edges
Et = En + Er

mdl = Model("Task Graph Problem")

#f is the flow across each edge
f = mdl.addVars(Et, lb = -4, ub = 4, vtype=GRB.INTEGER, name="f")
#p is the flow component of the reward function for each edge
p = mdl.addVars(Ereg, lb = -10000, ub = 10000, vtype=GRB.INTEGER, name="p")
#d is the previous reward component of the reward function for each edge
d = mdl.addVars(Ereg, lb = -10000, ub = 10000, vtype=GRB.CONTINUOUS, name="d")
#r is the reward for each edge
r = mdl.addVars(En, lb = -10000, ub = 10000, vtype=GRB.CONTINUOUS, name="r")
#rc is the combined reward for edge [4,5] from [2,4] and [3,4]
rc = mdl.addVar(lb = -10000, ub = 10000, vtype=GRB.CONTINUOUS, name="rc")

mdl.modelSense = GRB.MINIMIZE

#objective is to maximize the total flow across edges
#in this case, the cost functions are negated because they are rewards
mdl.setObjective(
    quicksum(r[i,j] for i,j in En)
)

#flow cannot exceed edge capacity (4 because 4 robots)
#technically taken care of in variable lb and ub
mdl.addConstrs(f[i,j] <= 4 for i,j in En)

#flow over normal edges is inverse of flow on reverse edges
mdl.addConstrs(f[i,j] == -1*f[j,i] for i,j in En)

#flow conservation constraint
mdl.addConstrs(quicksum(f[i,j] for j in edges_tot[i]) == 0 for i in Vi)

#required flow constraints
#source (node 1) must have 4 outgoing flow
mdl.addConstr(quicksum(f[1,j] for j in edges_tot[1]) == 4)
#sink (node 5) must have 4 incoming flow
mdl.addConstr(quicksum(f[j,5] for j in edges_tot[5]) == 4)

#piecewise linear definition for diminishing reward on edge [1,2]
mdl.addGenConstrPWL(f[1,2], r[1,2], [-1,0,1], [-1,-1,-1])

#piecewise linear definition for diminishing reward on edge [1,3]a
mdl.addGenConstrPWL(f[1,3], r[1,3], [-1,0,1], [-1,-1,-1])

#PWLs for reward on edge [2,4]
mdl.addGenConstrPWL(f[2,4], p[2,4], [-1,0,1,2,3,4,5], [0,0,0,-5,-12,-13,-13])
mdl.addGenConstrPWL(r[1,2], d[2,4], [-20,0], [-1,-1])
mdl.addConstr(r[2,4] == p[2,4] + (ce[2,4]*d[2,4]))

#PWLs for reward on edge [3,4]
mdl.addGenConstrPWL(f[3,4], p[3,4], [-1,0,1,2,3,4,5], [0,0,0,-5,-5,-5,-5])
mdl.addGenConstrPWL(r[1,3], d[3,4], [-20,0], [-1,-1])
mdl.addConstr(r[3,4] == p[3,4] + (ce[3,4]*d[3,4]))

#rc is reward of [2,4] + reward of [3,4]
mdl.addConstr(rc == r[2,4] + r[3,4])

#PWLs for reward on edge [4,5]
mdl.addGenConstrPWL(f[4,5], p[4,5], [-1,0,1,2,3,4], [0,0,-3,-5,-6,-6])
mdl.addGenConstrPWL(rc, d[4,5], [-20,0], [-10,0])
mdl.addConstr(r[4,5] == p[4,5] + (ce[4,5]*d[4,5]))

#AND constraint on flows
mdl.addConstr(f[2,4] >= 1)
mdl.addConstr(f[3,4] >= 1)

mdl.params.MIPGap = 0.0001
mdl.params.Method = 5
mdl.params.TimeLimit = 30
mdl.params.MIPFocus = 1
mdl.optimize()

print("OBJECTIVE: ", mdl.objVal)

flow_dict = {}

for v in mdl.getVars():
    if(v.varName[0] == 'f'):
        print(v.varname, ":", v.x)
        if(v.x >= 0):
            flow_dict[(int(v.varname[2]), int(v.varname[4]))] = v.x

print("\n")

for v in mdl.getVars():
    if(v.varName[0] == 'r'):
        print(v.varname, ":", v.x)

print("\n")

for v in mdl.getVars():
    if(v.varName[0] == 'p'):
        print(v.varname, ":", v.x)

print("\n")

for v in mdl.getVars():
    if(v.varName[0] == 'd'):
        print(v.varname, ":", v.x)   

G = nx.DiGraph()
G.add_nodes_from(V)
G.add_edges_from(En)

color_map = []
for node in G:
    if(node == 1 or node == 6):
        color_map.append("green")
    else:
        color_map.append("yellow")

pos=nx.planar_layout(G)
nx.draw(G, pos, node_color=color_map, with_labels=True)
nx.draw_networkx_edge_labels(G, pos, edge_labels=flow_dict)
plt.axis('off')
plt.show()
