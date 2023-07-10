#!/usr/bin/env python
"""Advanced example"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from netgraph import Graph


fig, ax = plt.subplots(figsize=(6,6))

balanced_tree = nx.balanced_tree(3, 3)
g = Graph(
    balanced_tree,
    node_layout='radial',
    edge_layout='straight',
    node_color='crimson',
    node_size={node : 4 if node == 0 else 2 for node in balanced_tree}, # nearly all parameters can also be dictionaries
    node_edge_width=0,
    edge_color='black',
    edge_width=0.5,
    node_labels=dict(zip(balanced_tree, 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')),
    node_label_offset=0.05,
    node_label_fontdict=dict(fontsize=10),
    ax=ax,
)

# Note that Netgraph uses matplotlib for creating the visualisation.
# Node and edge artistis are derived from `matplotlib.patches.PathPatch`.
# Node and edge labels are `matplotlib.text.Text` instances.
# Hence all node artists, edge artists, and labels can be queried using the original
# node and edge identifiers, and then manipulated using standard matplotlib syntax.

# Example 1: center the label of the root node on the corresponding node artist and make it white.
root = 0
center = g.node_positions[root]
g.node_label_artists[root].set_position(center)
g.node_label_artists[root].set_color('white')

# Example 2: decrease the node artist alpha parameter from the root to the leaves or the graph:
for node in balanced_tree:
    distance = np.linalg.norm(center - g.node_positions[node])
    g.node_artists[node].set_alpha(1 - distance)

# Redraw figure to display changes
fig.canvas.draw()

fig.savefig("publication/advanced_example.png")
plt.show()
