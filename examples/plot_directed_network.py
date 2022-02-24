#!/usr/bin/env python
"""
Directed Graph
==============

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
    (7, 4),
    (0, 4),
    (1, 5), # <-
    (5, 1), # <-
    (2, 6),
    (3, 7)
]

Graph(cube, arrows=True)
plt.show()
