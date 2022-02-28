#!/usr/bin/env python
"""
Node and Edge Labels
====================

Labels
------

If the variables :code:`node_labels` and :code:`edge_labels` are set to :code:`True`,
the nodes and edges are labelled with the corresponding node and edge IDs.
Alternatively, :code:`node_labels` and :code:`edge_labels` can be
dictionaries that map node and edge IDs to custom strings.

Styling
-------

The contents of the variables :code:`node_label_fontdict` and :code:`edge_label_fontdict`
are passed to :code:`matplotlib.text.Text` to stylise the node label and edge label text objects.
Consult the :code:`matplotlib.text.Text` documentation for a full list of available options.
By default, the following values differ from the defaults for :code:`matplotlib.text.Text`:

- size (adjusted to fit into node artists if no :code:`node_label_offset` is used)
- horizontalalignment (default here: :code:`'center'`)
- verticalalignment (default here: :code:`'center'`)
- clip_on (default here: :code:`False`)
- zorder (default here: :code:`inf`)

Positioning
-----------

Edge labels are always centred on the corresponding edges.
The position of the edge label along the edge can be controlled using the
:code:`edge_label_position` argument:

- :code:`0.0` : edge labels are placed at the head of the edge
- :code:`0.5` : edge labels are placed at the centre of the edge (default)
- :code:`1.0` : edge labels are placed at the tail of the edge

If :code:`edge_label_rotate` is True (default), edge labels are rotated such
that they have the same orientation as their edge.
If False, edge labels are not rotated; the angle of the text is parallel to the axis.

By default, node labels are centred on the corresponding nodes.
However, they can also be offset using the :code:`node_label_offset` parameter.
If :code:`node_label_offset` is a (float dx, float dy) tuple,
node labels are offset by the corresponding amounts.
If :code:`node_label_offset` is a float, netgraph attempts to place node labels
within that distance from node centres while avoiding collisions with node and edges.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from netgraph import Graph

fig, (ax1, ax2) = plt.subplots(1, 2)

triangle = [(0, 1), (0, 2), (1, 1), (1, 2), (2, 0)]

node_positions = {
    0 : np.array([0.2, 0.2]),
    1 : np.array([0.5, 0.8]),
    2 : np.array([0.8, 0.2]),
}

g = Graph(
    triangle,
    node_layout=node_positions, edge_layout='curved', edge_layout_kwargs=dict(k=0.025),
    node_labels={0 : 'a', 1 : 'b', 2 : 'c'},
    edge_labels=True, edge_label_fontdict=dict(fontweight='bold'),
    ax=ax1,
)

h = Graph(nx.complete_graph(7), edge_width=0.5, node_labels=True,
      node_label_fontdict=dict(size=14), node_label_offset=0.075, ax=ax2)

################################################################################
# Node and edge label properties can also be changed individually after an
# initial draw using the standard :code:`matplotlib.text.Text` methods:

# make changes
g.edge_label_artists[(0, 1)].set_style('italic')
g.node_label_artists[1].set_color('hotpink')

# force redraw to display changes
fig.canvas.draw()
