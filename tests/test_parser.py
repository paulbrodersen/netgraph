#!/usr/bin/env python
"""
Test _parser.py.
"""

import numpy as np

from netgraph._main import Graph
from netgraph._parser import parse_graph


def test_edge_list():
    g = Graph([(0, 1)])
    assert g.nodes == [0, 1]
    assert g.edges == [(0, 1)]


def test_edge_list_with_weights():
    g = Graph([(0, 1, 0.5)])
    assert g.nodes == [0, 1]
    assert g.edges == [(0, 1)]


def test_sparse_matrix_format():
    g = Graph(np.array([[0, 1, 0.5]]))
    assert g.nodes == [0, 1]
    assert g.edges == [(0, 1)]


def test_full_rank_matrix_format():
    w = np.zeros((4, 4)) # (2, 2) or (3, 3) ndarrays are interpreted as edge lists!
    w[0, 1] = 0.5
    g = Graph(w)
    assert g.nodes == [0, 1, 2, 3]
    assert g.edges == [(0, 1)]


def test_networkx_graph():
    import networkx
    g = Graph(networkx.Graph([(0, 1)]))
    assert g.nodes == [0, 1]
    assert g.edges == [(0, 1)]


def test_igraph_graph():
    import igraph
    g = Graph(igraph.Graph([(0, 1)]))
    assert g.nodes == [0, 1]
    assert g.edges == [(0, 1)]


def test_graph_tool_graph():
    import graph_tool
    gt = graph_tool.Graph()
    v1 = gt.add_vertex()
    v2 = gt.add_vertex()
    e  = gt.add_edge(v1, v2)
    g = Graph(gt)
    assert g.nodes == [0, 1]
    assert g.edges == [(0, 1)]
