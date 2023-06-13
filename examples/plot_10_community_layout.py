#!/usr/bin/env python
"""
Community Node Layout / Bundled Edges
=====================================
"""

import matplotlib.pyplot as plt
import networkx as nx

from netgraph import Graph

# create a modular graph
partition_sizes = [10, 20, 30]
g = nx.random_partition_graph(partition_sizes, 0.5, 0.1)

# create a dictionary that maps nodes to the community they belong to
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
    3 : 'tab:red',
}
node_color = {node: community_to_color[community_id] for node, community_id in node_to_community.items()}

Graph(g,
      node_color=node_color, node_edge_width=0, edge_alpha=0.1,
      node_layout='community', node_layout_kwargs=dict(node_to_community=node_to_community),
      edge_layout='bundled', edge_layout_kwargs=dict(k=2000),
)

plt.show()

################################################################################
# Alternatively, the best partition into communities can be inferred, for example
# using the Louvain algorithm (:code:`pip install python-louvain`):

from community import community_louvain
node_to_community = community_louvain.best_partition(g)
