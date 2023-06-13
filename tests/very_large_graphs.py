#!/usr/bin/env python
"""
Test defensive measures against very large graphs.
"""

import matplotlib.pyplot as plt
import networkx as nx

from netgraph import Graph

g = nx.complete_graph(100)
# g = nx.erdos_renyi_graph(100, p=0.1)
Graph(g)
plt.show()
