#!/usr/bin/env python
"""
Flowchart
==========

Example adapted from https://xkcd.com/1195/.
"""

import matplotlib.pyplot as plt

from netgraph import InteractiveGraph as Graph

edges = [(0, 1), (1, 1)]

node_labels = {
    0 : "START",
    1 : "HEY, WAIT,\nTHIS FLOWCHART\nIS A TRAP!"
}

edge_labels = {
    (1, 1) : "YES"
}

node_shape = {
    0 : 's',
    1 : 'd',
}

node_size = {
    0 : 10,
    1 : 20,
}

node_positions = {
    0 : (0.5, 0.8),
    1 : (0.5, 0.3),
}

with plt.xkcd():
    fig, ax = plt.subplots()
    Graph(edges, node_layout=node_positions,
          node_labels=node_labels, edge_labels=edge_labels,
          node_size=node_size, node_shape=node_shape,
          arrows=True, ax=ax)
    ax.set_title("https://xkcd.com/1195/")
    plt.show()
