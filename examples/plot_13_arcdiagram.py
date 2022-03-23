#!/usr/bin/env python
"""
Arc Diagrams
============
"""

import matplotlib.pyplot as plt
import networkx as nx

from netgraph import ArcDiagram

# Create a modular graph.
partition_sizes = [5, 6, 7]
g = nx.random_partition_graph(partition_sizes, 1, 0.1)

# Create a dictionary that maps nodes to the community they belong to,
# and set the node colors accordingly.
node_to_community = dict()
node = 0
for community_id, size in enumerate(partition_sizes):
    for _ in range(size):
        node_to_community[node] = community_id
        node += 1

community_to_color = {
    0 : 'tab:blue',
    1 : 'tab:orange',
    2 : 'tab:green',
}
node_color = {node: community_to_color[community_id] for node, community_id in node_to_community.items()}

ArcDiagram(g, node_size=1, node_color=node_color, node_edge_width=0, edge_alpha=1., edge_width=0.1)
plt.show()

################################################################################
# By default, ArcDiagram optimises the node order such that the number of edge crossings is minimised.
# For larger graphs, this process can take a long time.
# The node order can be set explicitly using the :code:`node_order` argument.
# In this case, no optimisation is attempted.

ArcDiagram(g, node_order=range(len(g)), node_size=1, node_color=node_color, node_edge_width=0, edge_alpha=1., edge_width=0.1)
