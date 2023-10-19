#!/usr/bin/env python
"""
Curved Edges
============

In the 'curved' edge layout, edges are repelled by nodes. This reduces node/edge occlusions.
If the (optional) parameter :code:`bundle_parallel_edges` (default :code:`True`) is set to :code:`False`,
then edges also repel each other.
The :code:`k` parameter governs how strongly edges are displaced by these repulsive forces and hence curved.
"""

import numpy as np
import matplotlib.pyplot as plt

from netgraph import Graph

edges = [
    (0, 1),
    (1, 2),
    (0, 2),
    (2, 0),
]
node_positions = {
    0 : np.array([0.2, 0.5]),
    1 : np.array([0.5, 0.5]),
    2 : np.array([0.8, 0.5]),
}

Graph(edges,
      node_layout=node_positions,
      node_labels=True,
      edge_layout="curved",
      edge_layout_kwargs=dict(bundle_parallel_edges=False, k=0.1),
      arrows=True,
)
plt.show()
