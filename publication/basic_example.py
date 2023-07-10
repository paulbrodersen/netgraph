#!/usr/bin/env python
"""Basic example"""

import matplotlib.pyplot as plt

from netgraph import Graph

triangle = [
    (0, 1),
    (1, 2),
    (2, 0),
    (1, 1),
]
Graph(
    triangle,
    node_labels=True,
    arrows=True,
)
plt.gcf().savefig("publication/basic_example.png")
plt.show()
