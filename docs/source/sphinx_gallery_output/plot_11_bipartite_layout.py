#!/usr/bin/env python
"""
Bipartite node layout
=====================

"""

################################################################################
# By default, nodes are partitioned into two subsets using a two-coloring of the graph.
# The median heuristic proposed in Eades & Wormald (1994) is used to reduce edge crossings.

import matplotlib.pyplot as plt

from netgraph import Graph

edges = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (5, 6)
]

Graph(edges, node_layout='bipartite', node_labels=True)

plt.show()

################################################################################
# The partitions can also be made explicit using the :code:`subsets` argument.
# In multi-component bipartite graphs, multiple two-colorings are possible,
# such that explicit specification of the subsets may be necessary to achieve the desired partitioning of nodes.

import matplotlib.pyplot as plt

from netgraph import Graph

edges = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (5, 6)
]

Graph(edges, node_layout='bipartite', node_layout_kwargs=dict(subsets=[(0, 2, 4, 6), (1, 3, 5)]), node_labels=True)

plt.show()

################################################################################
# To change the layout from the left-right orientation to a bottom-up orientation,
# call the layout function directly and swap x and y coordinates of the node positions.

import matplotlib.pyplot as plt

from netgraph import Graph, get_bipartite_layout

edges = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (5, 6)
]

node_positions = get_bipartite_layout(edges, subsets=[(0, 2, 4, 6), (1, 3, 5)])
node_positions = {node : (x, y) for node, (y, x) in node_positions.items()}

Graph(edges, node_layout=node_positions, node_labels=True)

plt.show()
