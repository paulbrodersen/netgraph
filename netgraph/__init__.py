#!/usr/bin/env python
# -*- coding: utf-8 -*-

# netgraph.py --- Plot weighted, directed graphs of medium size (10-100 nodes).

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

Summary:
--------
Module to plot weighted, directed graphs of medium size (10-100 nodes).
Unweighted, undirected graphs will look perfectly fine, too, but this module
might be overkill for such a use case.

Raison d'être:
--------------
Existing draw routines for networks/graphs in python use fundamentally different
length units for different plot elements. This makes it hard to
    - provide a consistent layout for different axis / figure dimensions, and
    - judge the relative sizes of elements a priori.
This module amends these issues.

Furthermore, this module allows to tweak node positions using the
mouse after an initial draw.

Example:
--------
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import netgraph

# construct sparse, directed, weighted graph
# with positive and negative edges
n = 20
w = np.random.randn(n,n)
p = 0.2
c = np.random.rand(n,n) <= p
w[~c] = 0.

# plot
netgraph.draw(w)

# If no node positions are explicitly provided (via the `node_positions` argument to `draw`),
# netgraph uses a spring layout to position nodes (Fruchtermann-Reingold algorithm).
# If you would like to manually tweak the node positions using the mouse after the initial draw,
# use the InteractiveGraph class:

graph = netgraph.InteractiveGraph(w)

# The new node positions can afterwards be retrieved via:
pos = graph.node_positions

# IMPORTANT NOTE:
# You must retain a reference to the InteractiveGraph instance at all times (here `graph`).
# Otherwise, the object will be garbage collected and you won't be able to alter the node positions interactively.

"""

__version__ = "3.0.0"
__author__ = "Paul Brodersen"
__email__ = "paulbrodersen+netgraph@gmail.com"

from ._main import (draw,
                    draw_nodes,
                    draw_node_labels,
                    draw_edges,
                    draw_edge_labels,
                    parse_graph,
                    get_color,
                    fruchterman_reingold_layout,
                    spring_layout,
                    Graph,
                    InteractiveGraph,
                    test)

from ._interactive_variants import (InteractiveGrid,
                                    InteractiveHypergraph)

__all__ = ['draw',
           'draw_nodes',
           'draw_node_labels',
           'draw_edges',
           'draw_edge_labels',
           'parse_graph',
           'get_color',
           'fruchterman_reingold_layout',
           'spring_layout',
           'Graph',
           'InteractiveGraph',
           'InteractiveGrid',
           'InteractiveHypergraph',
           'test']
