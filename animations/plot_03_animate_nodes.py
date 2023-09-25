#!/usr/bin/env python
"""
Visualise Changes in Node Properties
====================================

Here, we visualise changes in the nodes of a network.
We change both, the colour and the size of the nodes.
"""
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from netgraph import Graph

# Simulate a dynamic network with
# - 5 frames / different node states,
# - with 10 nodes at each time point, and
# - an expected edge density of 25%.
total_frames = 5
total_nodes = 10
adjacency_matrix = np.random.rand(total_nodes, total_nodes) < 0.25
node_values = np.random.rand(total_frames, total_nodes)

cmap = plt.cm.viridis

fig, ax = plt.subplots()
g = Graph(adjacency_matrix,
          node_layout="circular",
          node_layout_kwargs=dict(reduce_edge_crossings=False),
          arrows=True,
          ax=ax
)

def update(ii):
    for node, artist in g.node_artists.items():
        value = node_values[ii, node]
        artist.set_facecolor(cmap(value))
        artist.set_edgecolor(cmap(value))
        # The default node size is 3., which is rescaled internally
        # to 0.03 to yield layouts comparable to networkx and igraph.
        # As the expectation of `value` is 0.5, we multiply `value` by 6 * 0.01,
        # and thus match the default node size on average.
        artist.radius = 6 * 0.01 * value
    return g.node_artists.values()

animation = FuncAnimation(fig, update, frames=total_frames, interval=200, blit=True)
