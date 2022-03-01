#!/usr/bin/env python
"""
Dynamic Networks
================

Visualise changes in edge weights over time.
"""
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from netgraph import Graph

# Simulate a dynamic network with
# - 5 frames / network states,
# - with 10 nodes at each time point,
# - an expected edge density of 25%, and
# - edge weights drawn from a Gaussian distribution.
total_nodes = 10
total_frames = 5
adjacency_matrix = np.random.rand(total_nodes, total_nodes) < 0.25
weight_matrix = np.random.randn(total_frames, total_nodes, total_nodes)

# Normalise the weights, such that they are on the interval [0, 1].
# They can then be passed directly to matplotlib colormaps (which expect floats on that interval).
vmin, vmax = -2, 2
weight_matrix[weight_matrix<vmin] = vmin
weight_matrix[weight_matrix>vmax] = vmax
weight_matrix -= vmin
weight_matrix /= vmax - vmin

cmap = plt.cm.RdGy

fig, ax = plt.subplots()
g = Graph(adjacency_matrix, edge_cmap=cmap, arrows=True, ax=ax)

def update(ii):
    artists = []
    for jj, kk in zip(*np.where(adjacency_matrix)):
        w = weight_matrix[ii, jj, kk]
        artist = g.edge_artists[(jj, kk)]
        artist.set_facecolor(cmap(w))
        artist.update_width(0.03 * np.abs(w-0.5)) # np.abs(w-0.5) so that large negative edges are also wide
        artists.append(artist)
    return artists

animation = FuncAnimation(fig, update, frames=total_frames, interval=200, blit=True)
