#!/usr/bin/env python
"""
Unweighted & Undirected Graphs (MWE)
=====================================

A minimal working example to visualise an unweighted graph using default parameters.
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

# Pass the graph data structure to netgraph.
Graph(cube)

# Raise the resulting matplotlib figure to display the results.
plt.show()
