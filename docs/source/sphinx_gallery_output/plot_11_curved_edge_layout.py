#!/usr/bin/env python
"""
Curved Edges
============

Curved edges are repelled by nodes and other edges. This reduces node/edge and edge/edge occlusions.
"""

import matplotlib.pyplot as plt
import networkx as nx

from netgraph import Graph

Graph(nx.wheel_graph(7), edge_layout='curved')
plt.show()
