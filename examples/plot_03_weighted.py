#!/usr/bin/env python
"""
Weighted Graphs (1)
===================

A minimal working example to visualise a weighted & directed graph using default parameters.
"""

import matplotlib.pyplot as plt

from netgraph import Graph

# Define the graph; here we use an edge list but we could also have used
# - a full rank matrix (numpy array) with more than one unique non-zero entry,
# - a networkx Graph object with edges with a 'weight' attribute, or
# - an igraph Graph object with edges with a 'weight' attribute.
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

Graph(weighted_cube, arrows=True)
plt.show()

################################################################################
# By default, if the graph appears to be weighted and the :code:`edge_color`
# parameter is left unspecified by the user, Netgraph represents different edge
# weights using different colors. The default colormap is :code:`'RdGy'`,
# but any diverging matplotlib colormap can be used:
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
# Note that Netgraph maps edge weights to colors such that a weight of zero
# always corresponds to the center of the color map.
# If all edge weights are strictly positive, weights are mapped to the
# left hand side of the color map with :code:`vmin=0` and :code:`vmax=np.max(weights)`.
# If edge weights are positive and negative, then the vmin and vmax boundaries
# are set to +/- the maximum absolute weight.
#
# Custom diverging colormaps can be created using matploltib's :code:`LinearSegmentedColormap`:

from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('my_name', ['red', 'white', 'blue'])
