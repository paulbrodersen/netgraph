#!/usr/bin/env python
"""
Changes in Connectivity
=======================

The simplest way to visualise changes in connectivity is to reveal or
hide nodes and edges as they are added or removed from the
network. Here we show how to manipulate the visibility of node and
edge artists to that effect.

"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from netgraph import Graph

# The house-X graph.
edges = [
    (0, 1),
    (1, 4),
    (4, 2),
    (2, 3),
    (3, 4),
    (4, 0),
    (0, 2),
    (2, 1),
]

node_layout = {
    0 : (0, 0),
    1 : (1, 0),
    2 : (1, 1),
    3 : (0.5, 1.5),
    4 : (0, 1),
}

# Initialize the drawing.
fig, ax = plt.subplots()
g = Graph(edges, node_layout=node_layout, ax=ax)

# In this example, we show one additional edge on each iteration/frame.
edge_visibility = {
    (0, 1) : [False, True, True, True, True, True, True, True, True],
    (1, 4) : [False, False, True, True, True, True, True, True, True],
    (4, 2) : [False, False, False, True, True, True, True, True, True],
    (2, 3) : [False, False, False, False, True, True, True, True, True],
    (3, 4) : [False, False, False, False, False, True, True, True, True],
    (4, 0) : [False, False, False, False, False, False, True, True, True],
    (0, 2) : [False, False, False, False, False, False, False, True, True],
    (2, 1) : [False, False, False, False, False, False, False, False, True],
}

# We only show nodes that are part of at least one visible edge.
node_visibility = {
    0 : [False, True, True, True, True, True, True, True, True],
    1 : [False, True, True, True, True, True, True, True, True],
    2 : [False, False, False, True, True, True, True, True, True],
    3 : [False, False, False, False, True, True, True, True, True],
    4 : [False, False, True, True, True, True, True, True, True],
}

def update(ii):
    stale_artists = []

    for edge, artist in g.edge_artists.items():
        artist.set_visible(edge_visibility[edge][ii])
        stale_artists.append(artist)

    for node, artist in g.node_artists.items():
        artist.set_visible(node_visibility[node][ii])
        stale_artists.append(artist)

    return stale_artists

animation = FuncAnimation(fig, update, frames=len(edges)+1, interval=200, blit=True)
