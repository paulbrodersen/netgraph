#!/usr/bin/env python
"""
Visualising Large Graphs
========================

Or: how to deal with hairballs
------------------------------

When visualising very large graphs, link diagrams often result in
so-called 'hairballs', in which nodes and edges are densely packed and
overlap considerably. This can make it difficult or impossible to discern
any structure in the graph.

"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from netgraph import Graph

Graph(nx.complete_graph(100), node_size=1, node_edge_width=0.1, edge_width=0.1)
plt.show()

################################################################################
# Often, the problem is not solvable by choosing a better node layout
# or edge routing algorithm. Nodes require a minimum size to be
# discernible, edges a minimum width, and both require some whitespace
# around them, so that they can be distinguished from other plot
# elements. On the other hand, figures and computer screens have a
# finite number pixels: with 10k edges and a large figure with
# dimensions 20 inches by 10 inches and 100 DPI, each edge can occupy
# at most 200 pixels before edges have to overlap. By comparison, each
# letter in this sentence is rendered using 200-400 pixels (depending
# on browser, font, fontsize, screen resolution, etc.).
#
# Remedies to this problem fall into three categories:
#
# 1. coarse-graining the graph,
# 2. visualising subgraphs,
# 3. visualising graph properties other than connectivity.
#
# 1. Coarse-graining
# ------------------
#
# Coarse-graining is a procedure in which groups of nodes are
# contracted into a single node and their edges merged. There are
# different coarse-graining procedures, and which one is the best for
# you will depend on the type of graph, its structure, and what
# properties of the graph your are interested in. However, often a
# rational choice to visualise the meso-scale in large graphs is to
# group nodes into communities, such that nodes are more densely
# connected with nodes within the same community than with nodes in
# other communities.

# create a modulat graph
partition_sizes = [50, 50, 100, 100, 200, 200, 300]
g = nx.random_partition_graph(partition_sizes, 0.1, 1e-4, seed=0)

# create a dictionary that maps nodes to the community they belong to
community_size = dict(list(enumerate(partition_sizes)))
node_to_community = dict()
node = 0
for community, size in community_size.items():
    for _ in range(size):
        node_to_community[node] = community
        node += 1

# compute the community structure of the graph
def get_community_graph(edges, node_to_community):
    """Convert the graph into a weighted network of communities."""
    community_edges = dict()
    for (n1, n2) in edges:
        c1 = node_to_community[n1]
        c2 = node_to_community[n2]
        if c1 != c2:
            community_edges[(c1, c2)] = community_edges.get((c1, c2), 0) + 1
    return community_edges

community_edges = get_community_graph(g.edges, node_to_community)

# plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.set_title("Original graph")
Graph(g, node_size=1, node_edge_width=0.1, edge_width=0.1, ax=ax1)

ax2.set_title("Coarse-grained graph structure")
edges = list(community_edges.keys())
node_size = {community : size / 50 for (community, size) in community_size.items()}
edge_width = {edge : np.log(weight + 1) for edge, weight in community_edges.items()}
Graph(edges, node_size=node_size, edge_width=edge_width, ax=ax2)

plt.show()

################################################################################
# If the communities are unknown, they can be inferred, for example
# using the Louvain algorithm (:code:`pip install python-louvain`):

from community import community_louvain
node_to_community = community_louvain.best_partition(g)

################################################################################
# 2. Visualising subgraphs
# ------------------------
#
# If the graph consists of multiple components, i.e. multiple
# connected subgraphs that are not part of any larger connected
# subgraph, the simplest solution is to plot each component separately.

# create a multi-component graph
g = nx.Graph()
for ii in range(3):
    h = nx.erdos_renyi_graph(100, 0.05)
    g = nx.disjoint_union(g, h)

# plot components
for ii, component in enumerate(nx.components.connected_components(g)):
    subgraph = nx.subgraph(g, component)
    fig, ax = plt.subplots()
    Graph(subgraph, node_size=1, node_edge_width=0.1, edge_width=0.1, ax=ax)
    ax.set_title(f"Component {ii+1}")
plt.show()

################################################################################
# If there some nodes are of particular interest, for example, because
# they are densely connected hubs, it can be instructive to visualize
# their immediate vicinity.

hub_node = sorted(g, key=lambda x: g.degree(x), reverse=True)[0]
h = nx.ego_graph(g, hub_node, radius=3)
node_color = {node : "tab:red" if node == hub_node else "white" for node in h.nodes}
node_size = {node : 5 if node == hub_node else 1 for node in h.nodes}
fig, ax = plt.subplots()
Graph(h, node_color=node_color, node_size=node_size, node_edge_width=0.1, edge_width=0.1, ax=ax)
ax.set_title(f"Vicinity of Hub Node {hub_node}")
plt.show()

################################################################################
# 3. Visualising other graph properties
# -------------------------------------
#
# Even if the connectivity of a graph cannot be visualised, other
# properties of the graph often can be, such as the degree
# distribution, the various centrality measures, assortativity, and
# clustering, to name but a few.
#
# Recent advances to characterise the local neighbourhood of nodes in
# graphs with multiple node types (e.g. graphs corresponding to
# molecules composed of different elements) are graph embeddings such
# as node2vec_, DeepWalk_, and `graph convolutions / Graph Neural
# Networks (GNNs)`_. The aim of these methods is to systematically
# describe each node and its neighbourhood using only a small, ordered
# set of floats. Each such set forms a point in a multi-dimensional
# space (i.e. the embedding), and points that are close in this space
# correspond to nodes with similar properties and neighbourhoods.  By
# reducing the dimensionality of the embedding down to two dimensions
# using standard techniques such as PCA or UMAP, or by simply plotting
# only two dimensions at a time, these node embeddings can be readily
# visualised. However, creating such visualisations is outside the
# scope of netgraph, which is a library to create link-diagrams.
#
# .. _node2vec: https://distill.pub/2021/understanding-gnns/
# .. _DeepWalk: https://github.com/phanein/deepwalk
# .. _graph convolutions / Graph Neural Networks (GNNs): https://distill.pub/2021/understanding-gnns/
