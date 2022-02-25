#!/usr/bin/env python
"""
Circular node layout
====================

The circular node layout routine in netgraph uses the Baur-Brandes algorithm to reduce edge crossings.
"""

import matplotlib.pyplot as plt

from netgraph import Graph

unbalanced_tree = [
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (2, 6),
    (3, 7),
    (3, 8),
    (4, 9),
    (4, 10),
    (4, 11),
    (5, 12),
    (5, 13),
    (5, 14),
    (5, 15)
]

Graph(unbalanced_tree, node_layout='circular')

plt.show()
