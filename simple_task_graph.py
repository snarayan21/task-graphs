import networkx as nx
from networkx.algorithms.flow import maximum_flow
import matplotlib.pyplot as plt

G = nx.DiGraph()

G.add_node("a", demand=-4)
G.add_node("g", demand=4)
G.add_edge("a", "b", capacity=4, weight=-1)
G.add_edge("b", "c", capacity=2, weight=-1)
G.add_edge("c", "d", capacity=4, weight=-1)
G.add_edge("a", "e", capacity=4, weight=-1)
G.add_edge("e", "f", capacity=4, weight=-1)
G.add_edge("f", "g", capacity=4, weight=-1)
G.add_edge("d", "g", capacity=4, weight=-1)

mincostFlow = nx.min_cost_flow(G)
mincost = nx.cost_of_flow(G, mincostFlow)
print(mincost)
print(mincostFlow)
label_edges = {}
for k1,v1 in mincostFlow.items():
    for k2,v2 in v1.items():
        label_edges[(k1, k2)] = (G[k1][k2]["capacity"], G[k1][k2]["weight"], v2)
print(label_edges)

pos=nx.planar_layout(G)
nx.draw(G, pos, with_labels=True)
nx.draw_networkx_edge_labels(G, pos, edge_labels=label_edges)
plt.axis('off')
plt.show()