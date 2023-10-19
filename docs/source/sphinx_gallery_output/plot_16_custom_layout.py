#!/usr/bin/env python
"""
Custom Node Positions and Edge Paths
====================================

Typically, node and edge layouts are specified using a strings,
e.g. with :code:`node_layout='spring', edge_layout='curved'`.
However, node positions can be set explicitly by using a dictionary that maps
node IDs to (x, y) coordinates as the :code:`node_layout` keyword argument.
Analogously, edge paths can be set explicitly by using a dictionary that maps
edge IDs to ndarray of (x, y) coordinates as the :code:`edge_layout` keyword argument.
"""

import numpy as np
import matplotlib.pyplot as plt

from netgraph import Graph

edge_list = [(0, 1)]
node_positions = {
    0 : (0.2, 0.4),
    1 : (0.8, 0.6)
}
edge_paths = {
    (0, 1) : np.array([(0.2, 0.4), (0.2, 0.8), (0.5, 0.8), (0.5, 0.2), (0.8, 0.2), (0.8, 0.6)])
}

Graph(edge_list, node_layout=node_positions, edge_layout=edge_paths)
plt.show()
