#!/usr/bin/env python
"""
Weighted Graphs (1)
===================

Default visualisation for a weighted graph.
"""

import matplotlib.pyplot as plt

from netgraph import Graph

weighted_cube = [
    (0, 1, -0.1),
    (1, 2, -0.2),
    (2, 3, -0.4),
    (3, 2,  0.0),
    (3, 0, -0.2),
    (4, 5,  0.7),
    (5, 6,  0.9),
    (6, 5, -0.2),
    (6, 7,  0.5),
    (7, 4,  0.1),
    (0, 4,  0.5),
    (1, 5, -0.3),
    (5, 1, -0.4),
    (2, 6,  0.8),
    (3, 7,  0.4)
]

cmap = 'RdGy'
Graph(weighted_cube, edge_cmap=cmap, edge_width=2., arrows=True)
plt.show()

################################################################################
# By default, different edge weights are represented by different colors.
# The default colormap is :code:`'RdGy'` but any diverging matplotlib colormap can be used:
#
# - :code:`'PiYG'`
# - :code:`'PRGn'`
# - :code:`'BrBG'`
# - :code:`'PuOr'`
# - :code:`'RdGy'` (default)
# - :code:`'RdBu'`
# - :code:`'RdYlBu'`
# - :code:`'RdYlGn'`
# - :code:`'Spectral'`
# - :code:`'coolwarm'`
# - :code:`'bwr'`
# - :code:`'seismic'`
#
# If edge weights are strictly positive, weights are mapped to the
# left hand side of the color map with :code:`vmin=0` and :code:`vmax=np.max(weights)`.
# If edge weights are positive and negative, then weights are mapped to colors
# such that a weight of zero corresponds to the center of the color map;
# the boundaries are set to +/- the maximum absolute weight.
#
# Custom diverging colormaps can be created using matploltib's :code:`LinearSegmentedColormap`:

from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('my_name', ['red', 'white', 'blue'])
