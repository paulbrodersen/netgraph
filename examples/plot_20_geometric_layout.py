#!/usr/bin/env python
"""
Geometric node layout
=====================

Infer node positions given the length of the edges between them.

"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from netgraph import Graph

# create a random geometric graph
g = nx.random_geometric_graph(50, 0.3, seed=2)
original_positions = nx.get_node_attributes(g, 'pos')

fig, (ax1, ax2) = plt.subplots(1, 2)
plot_instance = Graph(g,
                      node_layout=original_positions,
                      node_size=1,
                      node_edge_width=0.1,
                      edge_width=0.1,
                      ax=ax1,
)
ax1.axis([0, 1, 0, 1])
ax1.set_title('Original node positions')

# compute edge lengths
edge_length = dict()
for (source, target) in g.edges:
    x1, y1 = original_positions[source]
    x2, y2 = original_positions[target]
    edge_length[(source, target)] = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# use non-linear optimisation to infer node positions given the edge lengths
Graph(g,
      node_layout="geometric",
      node_layout_kwargs=dict(edge_length=edge_length, tol=1e-3),
      node_size=1,
      node_edge_width=0.1,
      edge_width=0.1,
      ax=ax2,
)
ax2.axis([0, 1, 0, 1])
ax2.set_title('Reconstructed node positions')

plt.show()
