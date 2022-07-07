#!/usr/bin/env python
"""
Multi-partite & shell node layouts
==================================

Draw a multi-partite in successive layers or in concentric circles.
"""

################################################################################
# To draw a multi-partite graph in successive layers, use the :code:`multipartite` node layout.
# The :code:`layers` argument indicates in which layer each node is plotted, as well as the order of layers.

import numpy as np
import matplotlib.pyplot as plt

from netgraph import Graph

partitions = [
    list(range(3)),
    list(range(3, 9)),
    list(range(9, 21))
]

edges = list(zip(np.repeat(partitions[0], 2), partitions[1])) \
      + list(zip(np.repeat(partitions[0], 2), partitions[1][1:])) \
      + list(zip(np.repeat(partitions[1], 2), partitions[2])) \
      + list(zip(np.repeat(partitions[1], 2), partitions[2][1:]))

Graph(edges, node_layout='multipartite', node_layout_kwargs=dict(layers=partitions, reduce_edge_crossings=True), node_labels=True)

plt.show()

################################################################################
# To change the layout from the left-right orientation to a bottom-up orientation,
# call the layout function directly and swap x and y coordinates of the node positions.

import numpy as np
import matplotlib.pyplot as plt

from netgraph import Graph, get_multipartite_layout

partitions = [
    list(range(3)),
    list(range(3, 9)),
    list(range(9, 21))
]

edges = list(zip(np.repeat(partitions[0], 2), partitions[1])) \
      + list(zip(np.repeat(partitions[0], 2), partitions[1][1:])) \
      + list(zip(np.repeat(partitions[1], 2), partitions[2])) \
      + list(zip(np.repeat(partitions[1], 2), partitions[2][1:]))

node_positions = get_multipartite_layout(edges, partitions, reduce_edge_crossings=True)
node_positions = {node : (x, y) for node, (y, x) in node_positions.items()}

Graph(edges, node_layout=node_positions, node_labels=True)

plt.show()

################################################################################
# To draw a multi-partite graph in concentric circles, use the :code:`shell` node layout.
# The :code:`shells` argument indicates in which circle each node is plotted, as well as the order of shells.

import numpy as np
import matplotlib.pyplot as plt

from netgraph import Graph

partitions = [
    list(range(3)),
    list(range(3, 9)),
    list(range(9, 21))
]

edges = list(zip(np.repeat(partitions[0], 2), partitions[1])) \
      + list(zip(np.repeat(partitions[0], 2), partitions[1][1:])) \
      + list(zip(np.repeat(partitions[1], 2), partitions[2])) \
      + list(zip(np.repeat(partitions[1], 2), partitions[2][1:]))

Graph(edges, node_layout='shell', node_layout_kwargs=dict(shells=partitions, reduce_edge_crossings=True), node_labels=True)

plt.show()
