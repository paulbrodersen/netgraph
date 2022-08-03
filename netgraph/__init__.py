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
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from netgraph import Graph, InteractiveGraph
>>>
>>> # Several graph formats are supported:
>>> graph_data = [(0, 1), (1, 2), (2, 0)] # edge list
>>> # graph_data = [(0, 1, 0.2), (1, 2, -0.4), (2, 0, 0.7)] # edge list with weights
>>> # graph_data = np.random.rand(10, 10) # full rank matrix
>>> # graph_data = networkx.karate_club_graph() # networkx Graph/DiGraph objects
>>> # graph_data = igraph.Graph.Famous('Zachary') # igraph Graph objects
>>> # graph_data = graph_tool.collection.data['karate'] # graph_tool Graph objects
>>>
>>> # Create a non-interactive plot:
>>> Graph(graph_data)
>>> plt.show()
>>>
>>> # Create an interactive plot.
>>> # NOTE: you must retain a reference to the plot instance!
>>> # Otherwise, the plot instance will be garbage collected after the initial draw
>>> # and you won't be able to move the plot elements around.
>>> plt.ion()
>>> plot_instance = InteractiveGraph(graph_data)
>>> plt.show()
"""

__version__ = "4.9.1"
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
    'get_community_layout',
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
