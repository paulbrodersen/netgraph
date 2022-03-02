#!/usr/bin/env python
"""
Default Design
==============

Default visualisation for an unweighted graph.
"""

import matplotlib.pyplot as plt

from netgraph import Graph

cube = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (2, 6),
    (1, 5),
    (3, 7)
]

Graph(cube)
plt.show()
