from gurobipy import Model, GRB, quicksum, abs_, and_
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


#edges as adjacency list
#edges_n is normal directed graph edges
edges = {1: [2], 2: [3,5], 3: [4], 4: [6], 5: [6], 6: []}
#edges_tot is all graph edges connecting to some node
edges_tot = {1: [2], 2: [1,3,5], 3: [2,4], 4: [3,6], 5: [2,6], 6: [4,5]}

#sets needed
#V is the vertices
V = [i for i in range(1,7)]
#Vi is intermediate vertices
Vi = [i for i in range(2,6)]
#En is the normal directed graph edges
En = []
#En is the reverse directed graph edges
Er = []

Ereg = []
for i in range(1,7):
    conn = edges[i]
    for j in conn:
        En.append((i,j))
        Er.append((j,i))
        if(not((i == 3 and j == 4) or (i == 5 and j == 6))):
            print((i,j))
            Ereg.append((i,j))

#Et is both normal and reverse edges
Et = En + Er

mdl = Model("Task Graph Problem")

#f is the flow across each edge
f = mdl.addVars(Et, lb = -4, ub = 4, vtype=GRB.INTEGER, name="f")

mdl.modelSense = GRB.MINIMIZE

#objective is to minimize the cost of the total flow across edges
#in this case, the cost functions are negated because they are rewards
#all edges other than [3,4] and [5,6] have reward of 1 per unit flow
mdl.setObjective(
    quicksum(-1*f[i,j] for i,j in Ereg)
)

#flow cannot exceed edge capacity (4 because 4 robots)
#technically taken care of in variable lb and ub
mdl.addConstrs(f[i,j] <= 4 for i,j in En)

#flow over normal edges is inverse of flow on reverse edges
mdl.addConstrs(f[i,j] == -1*f[j,i] for i,j in En)

print(*V)

#flow conservation constraint
mdl.addConstrs(quicksum(f[i,j] for j in edges_tot[i]) == 0 for i in Vi)

#required flow constraints
#source (node 1) must have 4 outgoing flow
mdl.addConstr(quicksum(f[1,j] for j in edges_tot[1]) == 4)
#sink (node 6) must have 4 incoming flow
mdl.addConstr(quicksum(f[j,6] for j in edges_tot[6]) == 4)

#piecewise linear definition for reward of edge [3,4]
mdl.setPWLObj(f[3,4], [-1,0,1,2,4], [0,0,-2,-3,-3])

#piecewise linear definition for reward of edge [5,6]
mdl.setPWLObj(f[5,6], [0,2,3,4], [0,0,-5,-5])

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
