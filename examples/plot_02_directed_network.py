#!/usr/bin/env python
"""
Directed Graphs
===============

Default visualisation for a directed graph.
"""

import matplotlib.pyplot as plt

from netgraph import Graph

cube = [
    (0, 1),
    (1, 2),
    (2, 3), # <- bidirectional edges
    (3, 2), # <-
    (3, 0),
    (4, 5),
    (5, 6), # <-
    (6, 5), # <-
    (6, 7),
    (0, 4),
    (7, 4),
    (1, 5), # <-
    (5, 1), # <-
    (2, 6),
    (3, 7)
]

Graph(cube, edge_width=2., arrows=True)
plt.show()
