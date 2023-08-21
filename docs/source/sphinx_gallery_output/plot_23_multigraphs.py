#!/usr/bin/env python
"""Multigraphs
===========

A multigraph is a graph that is permitted to have more than one edge
with the same source and target nodes. Multigraphs are often used to
represent networks with distinct types of edges, such as, for example,
transportation networks that include different means of transportation.

This tutorial shows how to create a basic visualisation of a multigraph
and covers the different supported multigraph input formats.

"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import igraph

from netgraph import MultiGraph

# define the graph
edges = [
    (0, 0, "a"),
    (0, 1, "a"),
    (1, 1, "a"),
    (1, 1, "b"),
    (1, 2, "a"),
    (1, 2, "b"),
    (2, 0, "a"),
    (2, 0, "b"),
    (2, 0, "c"),
    (2, 2, "a"),
    (2, 2, "b"),
    (2, 2, "c"),
    (0, 3, "a"),
    (3, 0, "a"),
    (1, 3, "a"),
    (1, 3, "b"),
    (3, 1, "a"),
    (3, 1, "b"),
    (2, 3, "a"),
    (2, 3, "b"),
    (3, 2, "a"),
]

# color edges according to edge key
key_to_color = {
    "a" : "tab:blue",
    "b" : "tab:orange",
    "c" : "tab:red",
}
edge_color = {(source, target, key) : key_to_color[key] \
              for (source, target, key) in edges}

# plot
_ = MultiGraph(
    edges,
    edge_layout="curved",
    edge_color=edge_color,
    node_labels=True,
    arrows=True,
)

################################################################################
# Netgraph supports a variety of different input formats:

# 1. Edge lists:
# An iterable of (source node ID, target node ID, edge key) or
# (source node ID, target node ID, edge key, weight) tuples, or
# an equivalent (E, 3) or (E, 4)  numpy array (where E is the number of edges).
edges = [
    (0, 1, 0),
    (0, 1, 1),
    (0, 1, 2),
]
_ = MultiGraph(edges)

################################################################################
# 2. Adjacency matrices:
# A (V, V, L)  numpy array, where V is the number of nodes/vertices, and L is
# the number of layers. The absence of a connection is indicated by a zero.
adjacency = np.zeros((2, 2, 3))
adjacency[0, 1, 0] = 1
adjacency[0, 1, 1] = 1
adjacency[0, 1, 2] = 1
_ = MultiGraph(adjacency)

################################################################################
# 3. networkx.MultiGraph objects:
g = nx.MultiGraph(edges)
_ = MultiGraph(g)

################################################################################
# 4. igraph.Graph objects:
g = igraph.Graph([edge[:2] for edge in edges])
g.es["id"] = [edge[2] for edge in edges]
_ = MultiGraph(g)

################################################################################
# For technical reasons, :code:`graph_tool.Graph` multigraphs are not supported,
# even though plotting of :code:`graph_tool.Graph` objects is supported for non-multigraphs.
# Please convert your graph-tool multigraphs to any of the supported formats before plotting.
