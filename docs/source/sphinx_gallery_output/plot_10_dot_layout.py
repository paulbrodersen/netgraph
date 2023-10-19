#!/usr/bin/env python
"""
Dot and radial node layouts
===========================

Plot a tree or other directed, acyclic graph with the :code:`'dot'` or :code:`'radial'` node layout.
Netgraph uses an implementation of the Sugiyama algorithm provided by the grandalf_ library
(and thus does not require Graphviz to be installed).

.. _grandalf: https://github.com/bdcht/grandalf
"""

import matplotlib.pyplot as plt
import networkx as nx

from netgraph import Graph

unbalanced_tree = [
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (2, 6),
    (3, 7),
    (3, 8),
    (4, 9),
    (4, 10),
    (4, 11),
    (5, 12),
    (5, 13),
    (5, 14),
    (5, 15)
]

balanced_tree = nx.balanced_tree(3, 3)

fig, (ax1, ax2) = plt.subplots(1, 2)
Graph(unbalanced_tree, node_layout='dot', ax=ax1)
Graph(balanced_tree, node_layout='radial', ax=ax2)
plt.show()
