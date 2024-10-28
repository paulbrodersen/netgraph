#!/usr/bin/env python
"""
Custom Mouseover Highlighting
=============================

When using interactive graph classes (:py:class:`InteractiveGraph`, :py:class:`MutableGraph`, :py:class:`EditableGraph`, and the :py:class:`MultiGraph` and :py:class:`ArcDiagram` equivalents), hovering with the mouse over graph elements highlights them and other connected graph elements. By default, hovering over a node highlights the node, its neighbours, and any edges between them. Hovering over an edge highlights the edge, and its source and target node.

This behaviour can be changed by redefining the :code:`mouseover_hightlight_mapping` class attribute after instantiation.

.. note::
   The images in the documentation are static images that don't support interactive events. Please run the following example code locally to observe the intended behaviour.

"""

import matplotlib.pyplot as plt
from netgraph import InteractiveGraph

edges = [(0, 1), (1, 2), (2, 0)]
custom_mapping = {
    0 : [],                        # nothing is highlighted / everything is de-emphasized
    1 : [1],                       # only the node being hovered over is highlighted
    2 : [2, 1, 0, (1, 2), (2, 0)], # the node, its neighbours and connecting edges are highlighted (same as default behaviour)
    (0, 1) : [(2, 0)],             # another edge is highlighted
    (1, 2) : [(1, 2), 1, 2],       # the edge, and its source and target nodes are highlighted (same as default behaviour)
    # (2, 0)                       # nothing happens when hovering over edge (2, 0)
}

fig, ax = plt.subplots(figsize=(10, 10))
g = InteractiveGraph(edges, node_labels=True, edge_labels=True, ax=ax)
g.mouseover_highlight_mapping = custom_mapping
plt.show()

################################################################################
# To turn off highlighting completely, provide an empty dictionary instead:

fig, ax = plt.subplots(figsize=(10, 10))
g = InteractiveGraph(edges, node_labels=True, edge_labels=True, ax=ax)
g.mouseover_highlight_mapping = dict()
plt.show()
