#!/usr/bin/env python
"""
Weighted Graphs (2)
===================

An alternative visualisation for a weighted graph using edge widths to represent
the absolute edge weight, and colour to represent the sign of the weight.
"""

import matplotlib.pyplot as plt

from netgraph import Graph

weighted_cube = [
    (0, 1, -0.1),
    (1, 2, -0.2),
    (2, 3, -0.4),
    (3, 2,  0.0),
    (3, 0, -0.2),
    (4, 5,  0.7),
    (5, 6,  0.9),
    (6, 5, -0.2),
    (6, 7,  0.5),
    (7, 4,  0.1),
    (0, 4,  0.5),
    (1, 5, -0.3),
    (5, 1, -0.4),
    (2, 6,  0.8),
    (3, 7,  0.4)
]

edge_width = {(u, v) : 3 * abs(w) for (u, v, w) in weighted_cube}
edge_color = {(u, v) : 'blue' if w <= 0 else 'red' for (u, v, w) in weighted_cube}
Graph(weighted_cube, edge_width=edge_width, edge_color=edge_color, arrows=True)

plt.show()
