#!/usr/bin/env python
"""
Flowcharts
==========

While Netgraph is not designed to make flowcharts, it can be used for
that purpose. Simply define the shape of the nodes / text boxes using
matplotlib Path objects, and set their sizes and (initial) positions.
When using the :code:`InteractiveGraph` class, the node / text box
positions can be tweaked with the mouse after the initial draw. If the
font size is not set explicitly, Netgraph rescales the text until all
text objects fit into their node shape.

Example adapted from https://xkcd.com/1195/.

"""
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.path import Path
from netgraph import InteractiveGraph

edges = [(0, 1), (1, 1)]

node_labels = {
    0 : "START",
    1 : "HEY, WAIT,\nTHIS FLOWCHART\nIS A TRAP!"
}

edge_labels = {
    (1, 1) : "YES"
}

# Define the node shapes; note that the path objects have to be centred on zero and closed.
node_shape = {
    0 : Path([(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)], closed=True), # square
    1 : Path([(0, -1), (1, 0), (0, 1), (-1, 0), (0, -1)], closed=True), # diamond
}

node_size = {
    0 : 30,
    1 : 60,
}

node_positions = {
    0 : (0.5, 0.9),
    1 : (0.5, 0.3),
}

edge_layout = {
    (0, 1) : np.array([(0.5, 0.9), (0.5, 0.3)]),
    (1, 1) : np.array([(0.5, 0.3), (0.5, -0.2), (1.1, -0.2), (1.1, 0.3), (0.5, 0.3)]),
}

with plt.xkcd():
    fig, ax = plt.subplots()
    InteractiveGraph(edges,
                     node_layout=node_positions,
                     node_labels=node_labels,
                     node_shape=node_shape,
                     node_size=node_size,
                     # edge_layout="straight", edge_layout_kwargs=dict(selfloop_radius=0.3),
                     edge_layout=edge_layout,
                     edge_labels=edge_labels,
                     edge_label_position=0.35,
                     edge_width=2,
                     arrows=True,
                     ax=ax)
    ax.set_title("https://xkcd.com/1195/")
    plt.show()
