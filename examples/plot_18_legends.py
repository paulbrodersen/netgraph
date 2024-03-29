#!/usr/bin/env python
"""
Node and Edge Legends
=====================

Legends for node or edge properties can be created through the use of matplotlib proxy artists.
For a comprehensive guide, see the `matplotlib legend guide`_.

.. _matplotlib legend guide: https://matplotlib.org/stable/tutorials/intermediate/legend_guide.html#proxy-legend-handles

"""

import matplotlib.pyplot as plt

from netgraph import Graph

triangle = [(0, 1), (1, 2), (2, 0)]

node_positions = {
    0 : (0.2, 0.2),
    1 : (0.5, 0.8),
    2 : (0.8, 0.2),
}

node_color = {
    0 : 'tab:blue',
    1 : 'tab:orange',
    2 : 'tab:green',
}

node_shape = {
    0 : 's',
    1 : '8',
    2 : 'o',
}

node_size = {
    0 : 5,
    1 : 10,
    2 : 15,
}

edge_width = {
    (0, 1) : 1,
    (1, 2) : 2,
    (2, 0) : 3,
}

edge_color = {
    (0, 1) : 'tab:red',
    (1, 2) : 'tab:purple',
    (2, 0) : 'tab:brown'
}

fig, ax = plt.subplots()
g = Graph(
    triangle,
    node_layout=node_positions,
    node_labels=True,
    edge_labels=True,
    node_size=node_size,
    node_color=node_color,
    node_edge_color=node_color,
    node_shape=node_shape,
    edge_width=edge_width,
    edge_color=edge_color,
    ax=ax
)

# Create proxy artists for legend handles.

node_proxy_artists = []
for node in [0, 1, 2]:
    proxy = plt.Line2D(
        [], [],
        linestyle='None',
        color=node_color[node],
        marker=node_shape[node],
        markersize=node_size[node],
        label=node
    )
    node_proxy_artists.append(proxy)

node_legend = ax.legend(handles=node_proxy_artists, loc='upper left', title='Nodes')
ax.add_artist(node_legend)

edge_proxy_artists = []
for edge in triangle:
    proxy = plt.Line2D(
        [], [],
        linestyle='-',
        color=edge_color[edge],
        linewidth=edge_width[edge],
        label=edge
    )
    edge_proxy_artists.append(proxy)

edge_legend = ax.legend(handles=edge_proxy_artists, loc='upper right', title='Edges')
ax.add_artist(edge_legend)

plt.show()
