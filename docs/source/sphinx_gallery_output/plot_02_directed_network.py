#!/usr/bin/env python
"""
Directed Graphs (MWE)
=====================

A minimal working example to visualise a directed graph using default parameters.
"""

import matplotlib.pyplot as plt

from netgraph import Graph

# Define the graph; here we use an edge list but we could also have used
# - a full rank matrix (numpy array) with ones and zeros,
# - a networkx Graph object,
# - an igraph Graph object, or
# - a graph-tool Graph object.
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
    (5, 1), # <-
    (1, 5), # <-
    (2, 6),
    (3, 7)
]

# Set the `arrows` flag to true to indicate the directions of the edges.
Graph(cube, arrows=True)
plt.show()
