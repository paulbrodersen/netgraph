#!/usr/bin/env python
"""
Visualise Changes in Connectivity
=================================

Here, we demonstrate how to visualise changes in connectivity over time.
"""
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from netgraph import Graph

# Simulate a dynamic network with
# - 5 frames / network states,
# - with 10 nodes at each time point,
# - an expected edge density of 25% at each time point
total_frames = 5
total_nodes = 10
adjacency_matrix = np.random.rand(total_frames, total_nodes, total_nodes) < 0.25

fig, ax = plt.subplots()
g = Graph(np.ones((total_nodes, total_nodes)), edge_width=1.5, arrows=True, ax=ax) # initialise with fully connected graph

def update(ii):
    for (jj, kk), artist in g.edge_artists.items():
        # turn visibility of edge artists on or off, depending on the adjacency
        if adjacency_matrix[ii, jj, kk]:
            artist.set_visible(True)
        else:
            artist.set_visible(False)
    return g.edge_artists.values()

animation = FuncAnimation(fig, update, frames=total_frames, interval=200, blit=True)
