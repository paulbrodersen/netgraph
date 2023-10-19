#!/usr/bin/env python
"""
Arc Diagrams
============
"""

import matplotlib.pyplot as plt
import networkx as nx

from netgraph import ArcDiagram

n = 20
g = nx.random_tree(n, seed=42)

ArcDiagram(g, node_labels=True, node_size=1, node_edge_width=0.1, edge_width=0.1)
plt.show()

################################################################################
# By default, ArcDiagram optimises the node order such that the number of edge crossings is minimised.
# For larger graphs, this process can take a long time.
# The node order can be set explicitly using the :code:`node_order` argument.
# In this case, no optimisation is attempted.

ArcDiagram(g, node_order=range(n), node_labels=True, node_size=1, node_edge_width=0.1, edge_width=0.1)
plt.show()

################################################################################
# Arc diagrams are useful to compare two networks with the same number of nodes,
# for example, the connectivity in the same network before and after some changes.

# Swap a few edges in the original graph. As this occurs in-place, we first make a copy.
h = g.copy()
nx.double_edge_swap(h, nswap=3)

# Visualise the changes in connectivity by plotting the two configurations above and below the center line.
# Highlight edges that were removed in red; new edges are shown in blue.
fig, ax = plt.subplots()
edge_color = {edge : "tab:red" if edge not in h.edges() else "lightgray" for edge in g.edges()}
ArcDiagram(g, above=False, node_order=range(n), node_size=1, node_edge_width=0.1, edge_color=edge_color, edge_width=0.5, ax=ax)
edge_color = {edge : "tab:blue" if edge not in g.edges() else "lightgray" for edge in h.edges()}
ArcDiagram(h, above=True,  node_order=range(n), node_size=1, node_edge_width=0.1, edge_color=edge_color, edge_width=0.5, ax=ax)
plt.show()
