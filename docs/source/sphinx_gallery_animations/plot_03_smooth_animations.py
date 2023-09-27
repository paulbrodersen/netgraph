#!/usr/bin/env python
"""
Smooth Animations
=================

For dynamic networks, the available data is often quite granular, and the large
differences between successive network states can result in janky animations.
Here we show how to interpolate between network states to obtain a smoother
presentation of the data.
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import TwoSlopeNorm
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

# Simulate a series of network states, such that each edge is assigned a different random value for each successive state.
total_states = 10
edge_values = {edge : np.random.normal(0, 1, size=total_states) for edge in edges}
states = np.arange(total_states)
total_interpolated_states = 100
interpolated_states = np.linspace(states.min(), states.max(), total_interpolated_states)
interpolated_edge_values = {edge : np.interp(interpolated_states, states, values) for edge, values in edge_values.items()}

# Map positive and negative edge values to diverging colors; ensure that zero is the center of the colormap.
norm = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)

def get_color(v):
    return plt.cm.coolwarm(norm(v))

# Map large absolute edge values (including negative ones) to large edge widths.
def get_width(v, max_width=3):
    width = 2 * abs(norm(v) - 0.5)
    # Ensure that the returned edge width is non-zero.
    return max(max_width * width, 1e-3)

def update(ii):
    stale_artists = []
    for (edge, artist) in g.edge_artists.items():
        value = interpolated_edge_values[edge][ii]
        artist.set_color(get_color(value))
        # Normally, netgraph re-scales edge widths internally by BASE_SCALE to ensure
        # that "typical" edge widths look comparable in netgraph, networkx, and igraph.
        # As we are manipulating the edge artist directly, we have to do the same here.
        artist.update_width(get_width(value) * BASE_SCALE)
        stale_artists.append(artist)
    return stale_artists

animation = FuncAnimation(fig, update, frames=total_interpolated_states, interval=50, blit=True)
