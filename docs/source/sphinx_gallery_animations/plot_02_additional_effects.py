#!/usr/bin/env python
"""
Changes in Node and Edge Properties
===================================

Here we show how to manipulate the color, transparency, and extent of node and
edge artists to visualize changes in node and edge properties.

"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from netgraph import Graph, BASE_SCALE

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

# On each iteration, we increase the emphasis of one additional edge
# by making it wider, darker, and making it less transparent.
# Nodes that are part of an emphasized edge undergo the same treatment.
edge_emphasis = {
    (0, 1) : [False, True, True, True, True, True, True, True, True],
    (1, 4) : [False, False, True, True, True, True, True, True, True],
    (4, 2) : [False, False, False, True, True, True, True, True, True],
    (2, 3) : [False, False, False, False, True, True, True, True, True],
    (3, 4) : [False, False, False, False, False, True, True, True, True],
    (4, 0) : [False, False, False, False, False, False, True, True, True],
    (0, 2) : [False, False, False, False, False, False, False, True, True],
    (2, 1) : [False, False, False, False, False, False, False, False, True],
}
edge_color = {edge : ["midnightblue" if emphasis else "lightblue" for emphasis in values] for edge, values in edge_emphasis.items()}
edge_alpha = {edge : [1.0 if emphasis else 0.5 for emphasis in values] for edge, values in edge_emphasis.items()}
edge_width = {edge : [1.5 if emphasis else 0.5 for emphasis in values] for edge, values in edge_emphasis.items()}

node_emphasis = {
    0 : [False, True, True, True, True, True, True, True, True],
    1 : [False, True, True, True, True, True, True, True, True],
    2 : [False, False, False, True, True, True, True, True, True],
    3 : [False, False, False, False, True, True, True, True, True],
    4 : [False, False, True, True, True, True, True, True, True],
}
node_color = {node : ["midnightblue" if emphasis else "lightblue" for emphasis in values] for node, values in node_emphasis.items()}
node_alpha = {node : [1.0 if emphasis else 0.5 for emphasis in values] for node, values in node_emphasis.items()}
node_size  = {node : [2.0 if emphasis else 1.0 for emphasis in values] for node, values in node_emphasis.items()}

def update(ii):
    stale_artists = []

    for edge, artist in g.edge_artists.items():
        artist.set_color(edge_color[edge][ii])
        artist.set_alpha(edge_alpha[edge][ii])
        # Normally, netgraph re-scales edge widths internally by BASE_SCALE to ensure
        # that "typical" edge widths look comparable in netgraph, networkx, and igraph.
        # As we are manipulating the edge artist directly, we have to do the same here.
        artist.update_width(edge_width[edge][ii] * BASE_SCALE)
        stale_artists.append(artist)

    for node, artist in g.node_artists.items():
        artist.set_color(node_color[node][ii])
        artist.set_alpha(node_alpha[node][ii])
        # Ditto for node sizes.
        artist.size = node_size[node][ii] * BASE_SCALE
        stale_artists.append(artist)

    return stale_artists

animation = FuncAnimation(fig, update, frames=len(edges)+1, interval=200, blit=True)
