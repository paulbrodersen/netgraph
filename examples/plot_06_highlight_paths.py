#!/usr/bin/env python
"""
Highlight paths
===============

"""


import matplotlib.pyplot as plt
import networkx as nx

from netgraph import Graph

# create a random geometric graph and plot it
g = nx.random_geometric_graph(100, 0.15, seed=42)
node_positions = nx.get_node_attributes(g, 'pos')
plot_instance = Graph(g,
                      node_layout=node_positions,
                      node_size=1,
                      node_edge_width=0.1,
                      edge_width=0.1)

# select a random path in the network and plot it
path = nx.shortest_path(g, 33, 66)

for node in path:
    node_artist = plot_instance.node_artists[node]
    node_artist.size *= 2
    node_artist.set_color('orange')

for ii, node_1 in enumerate(path[:-1]):
    node_2 = path[ii+1]
    if (node_1, node_2) in plot_instance.edges:
        edge = (node_1, node_2)
    else: # the edge is specified in reverse node order
        edge = (node_2, node_1)
    edge_artist = plot_instance.edge_artists[edge]
    edge_artist.update_width(2 * edge_artist.width)
    edge_artist.set_color('red')
    edge_artist.set_alpha(1.0)

plt.show()
