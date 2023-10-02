#!/usr/bin/env python
"""
Community Node Layout / Edge Bundling
=====================================

Networks often contain groups of nodes, which share some property and
are typically more densely connected with each other than with nodes outside of that group.
A typical example are friendship groups within the wider facebook graph of acquaintances.

Here we emphasise the modular structure of a graph by using the the :code:`community` node layout,
which places nodes belonging to the same community close to each other,
by giving nodes from the same community the same colour, and
by using edge bundling, which tends to de-emphasize the connectivity between different parts of a network.
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
}
node_color = {node: community_to_color[community_id] for node, community_id in node_to_community.items()}

Graph(g,
      node_color=node_color, node_edge_width=0, edge_alpha=0.1,
      node_layout='community', node_layout_kwargs=dict(node_to_community=node_to_community),
      edge_layout='bundled', edge_layout_kwargs=dict(k=2000, verbose=False),
)

plt.show()

################################################################################
# Unlike in the example above, the community structure of a graph is often not known but has to be inferred.
# For an overview of available community detection algorithms, see `Porter et al. (2009) Communities in networks`.
# Community detection algorithms are outside the scope of the Netgraph library,
# but many are available in networkx_, igraph_, and graph-tool_.
# Do note that community detection has been a rapidly evolving field and
# that classic algorithms such as Newman-Girvan or Louvain may not be the best available choices nowadays.
# For an opinionated but very readable introduction into recent developments in this area,
# see the blog_ of Tiago de Paula Peixoto, the author of the graph-tool library.
#
# .. _Porter et al. (2009) Communities in networks: https://doi.org/10.48550/arXiv.0902.3788
# .. _networkx: https://networkx.org/documentation/stable/reference/algorithms/community.html
# .. _igraph: https://igraph.org/c/doc/igraph-Community.html
# .. _graph-tool: https://graph-tool.skewed.de/static/doc/demos/inference/inference.html
# .. _blog: https://skewed.de/tiago/blog/modularity-harmful
