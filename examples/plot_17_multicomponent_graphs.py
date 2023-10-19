#!/usr/bin/env python
"""
Graphs with Multiple Components
===============================

Many node layout algorithms are only properly defined for connected graphs,
and yield bad results when applied to graphs with multiple components.
To circumvent this issue, Netgraph computes the node layout separately for each component,
and then arranges the individual components with respect to each other using `rectangle packing`__.

.. __ : https://github.com/Penlect/rectangle-packer

"""

import matplotlib.pyplot as plt

from itertools import combinations
from netgraph import Graph

edge_list = []

# add 15 2-node components
edge_list.extend([(ii, ii+1) for ii in range(30, 60, 2)])

# add 10 3-node components
for ii in range(60, 90, 3):
    edge_list.extend([(ii, ii+1), (ii, ii+2), (ii+1, ii+2)])

# add a couple of larger components
n = 90
for ii in [10, 20, 30]:
    edge_list.extend(list(combinations(range(n, n+ii), 2)))
    n += ii

nodes = list(range(n))
Graph(edge_list, nodes=nodes, node_size=1, edge_width=0.3, node_layout='circular')
plt.show()
