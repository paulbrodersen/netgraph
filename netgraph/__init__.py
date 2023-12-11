#!/usr/bin/env python
# -*- coding: utf-8 -*-

# netgraph.py --- Python drawing utilities for publication quality plots of networks.

# Copyright (C) 2016 Paul Brodersen <paulbrodersen+netgraph@gmail.com>

# Author: Paul Brodersen <paulbrodersen+netgraph@gmail.com>

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""
Netgraph
========

Summary
-------
Python drawing utilities for publication quality plots of networks.

Examples
--------
>>> import matplotlib.pyplot as plt
>>> from netgraph import Graph, InteractiveGraph, EditableGraph
>>>
>>> # Several graph formats are supported:
>>>
>>> # 1) edge lists
>>> graph_data = [(0, 1), (1, 2), (2, 0)]
>>>
>>> # 2) edge list with weights
>>> graph_data = [(0, 1, 0.2), (1, 2, -0.4), (2, 0, 0.7)]
>>>
>>> # 3) full rank matrices
>>> import numpy
>>> graph_data = np.random.rand(10, 10)
>>>
>>> # 4) networkx Graph and DiGraph objects (MultiGraph objects are not supported, yet)
>>> import networkx
>>> graph_data = networkx.karate_club_graph()
>>>
>>> # 5) igraph.Graph objects
>>> import igraph
>>> graph_data = igraph.Graph.Famous('Zachary')
>>>
>>> # 6) graph_tool.Graph objects
>>> import graph_tool.collection
>>> graph_data = graph_tool.collection.data["karate"]
>>>
>>> # Create a non-interactive plot:
>>> Graph(graph_data)
>>> plt.show()
>>>
>>> # Create an interactive plot, in which the nodes can be re-positioned with the mouse.
>>> # NOTE: you must retain a reference to the plot instance!
>>> # Otherwise, the plot instance will be garbage collected after the initial draw
>>> # and you won't be able to move the plot elements around.
>>> # For related reasons, if you are using PyCharm, you have to execute the code in
>>> # a console (Alt+Shift+E).
>>> plot_instance = InteractiveGraph(graph_data)
>>> plt.show()
>>>
>>> # Create an editable plot, which is an interactive plot with the additions
>>> # that nodes and edges can be inserted or deleted, and labels and annotations
>>> # can be created, edited, or deleted as well.
>>> plot_instance = EditableGraph(graph_data)
>>> plt.show()
>>>
>>> # Netgraph uses matplotlib for creating the visualisation.
>>> # Node and edge artistis are derived from `matplotlib.patches.PathPatch`.
>>> # Node and edge labels are `matplotlib.text.Text` instances.
>>> # Standard matplotlib syntax applies.
>>> fig, ax = plt.subplots(figsize=(5,4))
>>> plot_instance = Graph([(0, 1)], node_labels=True, edge_labels=True, ax=ax)
>>> plot_instance.node_artists[0].set_alpha(0.2)
>>> plot_instance.edge_artists[(0, 1)].set_facecolor('red')
>>> plot_instance.edge_label_artists[(0, 1)].set_style('italic')
>>> plot_instance.node_label_artists[1].set_size(10)
>>> ax.set_title("This is my fancy title.")
>>> ax.set_facecolor('honeydew') # change background color
>>> fig.canvas.draw() # force redraw to display changes
>>> fig.savefig('test.pdf', dpi=300)
>>> plt.show()
>>>
>>> # Read the documentation for a full list of available arguments:
>>> help(Graph)
>>> help(InteractiveGraph)
>>> help(EditableGraph)
"""

__version__ = "4.13.2"
__author__ = "Paul Brodersen"
__email__ = "paulbrodersen+netgraph@gmail.com"

from ._main import (
    BaseGraph,
    Graph,
    InteractiveGraph,
)

from ._node_layout import (
    get_random_layout,
    get_fruchterman_reingold_layout,
    get_sugiyama_layout,
    get_radial_tree_layout,
    get_circular_layout,
    get_linear_layout,
    get_bipartite_layout,
    get_multipartite_layout,
    get_shell_layout,
    get_community_layout,
    get_geometric_layout,
)

from ._edge_layout import (
    get_straight_edge_paths,
    get_curved_edge_paths,
    get_arced_edge_paths,
    get_bundled_edge_paths,
)

from ._interactive_variants import (
    MutableGraph,
    EditableGraph,
)

from ._arcdiagram import (
    BaseArcDiagram,
    ArcDiagram,
    InteractiveArcDiagram,
    EditableArcDiagram,
)

from ._parser import parse_graph


__all__ = [
    'get_random_layout',
    'get_fruchterman_reingold_layout',
    'get_sugiyama_layout',
    'get_radial_tree_layout',
    'get_circular_layout',
    'get_linear_layout',
    'get_bipartite_layout',
    'get_multipartite_layout',
    'get_community_layout',
    'get_geometric_layout',
    'get_straight_edge_paths',
    'get_curved_edge_paths',
    'get_bundled_edge_paths',
    'get_arced_edge_paths',
    'Basegraph',
    'Graph',
    'InteractiveGraph',
    'MutableGraph',
    'EditableGraph',
    'parse_graph',
    'ArcDiagram',
    'InteractiveArcDiagram',
    'MutableArcDiagram',
    'EditableArcDiagram',
]
