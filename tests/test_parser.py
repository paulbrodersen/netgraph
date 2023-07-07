#!/usr/bin/env python
"""
Test _parser.py.
"""

import numpy as np

from netgraph._parser import (
    _parse_edge_list,
    _parse_sparse_matrix_format,
    _parse_adjacency_matrix,
    _parse_nparray,
    _parse_networkx_graph,
    _parse_igraph_graph,
    _parse_graph_tool_graph,
    parse_graph,
)


def test_parse_edge_list():
    # 1) plain edge list as iterable of tuples
    provided_edges = [
        (0, 0),
        (0, 1),
    ]
    returned_edges = _parse_edge_list(provided_edges)
    for edge in provided_edges:
        assert edge in returned_edges

    # 2) edge list as array
    provided_edges = np.array([
        (0, 0),
        (0, 1),
    ])
    returned_edges = _parse_edge_list(provided_edges)
    for edge in provided_edges:
        assert tuple(edge) in returned_edges


def test_parse_sparse_matrix_format():
    # 1) unweighted
    provided_edges = np.array([
        (0, 0),
        (0, 1),
    ])
    returned_nodes, returned_edges, returned_weights = \
        _parse_sparse_matrix_format(provided_edges)
    for edge in provided_edges:
        assert tuple(edge) in returned_edges
    assert {0, 1} == set(returned_nodes)
    assert returned_weights is None

    # 2) weighted
    provided_edges = np.array([
        (0, 0, 0.),
        (0, 1, 1.),
    ])
    returned_nodes, returned_edges, returned_weights = \
        _parse_sparse_matrix_format(provided_edges)
    for edge in provided_edges:
        assert tuple(edge[:2]) in returned_edges
    assert {0, 1} == set(returned_nodes)
    assert {(0, 0), (0, 1)} == set(returned_weights.keys())
    assert {0., 1.} == set(returned_weights.values())

    # 3) multi-graph
    provided_edges = np.array([
        (0, 0),
        (0, 1),
        (0, 1),
    ])
    returned_nodes, returned_edges, returned_weights = \
        _parse_sparse_matrix_format(provided_edges)
    for edge in provided_edges:
        assert tuple(edge) in returned_edges
    assert {0, 1} == set(returned_nodes)
    assert {(0, 0), (0, 1)} == set(returned_weights.keys())
    assert {1, 2} == set(returned_weights.values())


def test_parse_adjacency_matrix():
    # 1) unweighted
    provided_adjacency = np.array([
        [1, 1],
        [0, 0]
    ])
    returned_nodes, returned_edges, returned_weights = \
        _parse_adjacency_matrix(provided_adjacency)
    desired_edges = [
        (0, 0),
        (0, 1),
    ]
    for edge in desired_edges:
        assert edge in returned_edges
    assert {0, 1} == set(returned_nodes)
    assert returned_weights is None

    # 2) weighted
    provided_adjacency = np.array([
        [1, 2],
        [0, 0]
    ])
    returned_nodes, returned_edges, returned_weights = \
        _parse_adjacency_matrix(provided_adjacency)
    desired_edges = [
        (0, 0),
        (0, 1),
    ]
    for edge in desired_edges:
        assert edge in returned_edges
    assert {0, 1} == set(returned_nodes)
    assert {(0, 0), (0, 1)} == set(returned_weights.keys())
    assert {1, 2} == set(returned_weights.values())


def test_parse_nparray():
    # 1) sparse matrix
    provided_edges = np.array([
        (0, 0),
        (0, 1),
    ])
    returned_nodes, returned_edges, returned_weights = \
        _parse_nparray(provided_edges)
    for edge in provided_edges:
        assert tuple(edge) in returned_edges
    assert {0, 1} == set(returned_nodes)
    assert returned_weights is None

    # 2) full-rank adjacency matrix
    provided_adjacency = np.array([
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    returned_nodes, returned_edges, returned_weights = \
        _parse_nparray(provided_adjacency)
    desired_edges = [
        (0, 0),
        (0, 1),
    ]
    for edge in desired_edges:
        assert edge in returned_edges
    assert {0, 1, 2, 3} == set(returned_nodes)
    assert returned_weights is None


def test_parse_networkx_graph():
    import networkx
    # 1) unweighted
    provided_edges = [
        (0, 0),
        (0, 1),
    ]
    g = networkx.Graph(provided_edges)
    returned_nodes, returned_edges, returned_weights = \
        _parse_networkx_graph(g)
    for edge in provided_edges:
        assert edge in returned_edges
    assert {0, 1} == set(returned_nodes)
    assert returned_weights is None

    # 2) weighted
    provided_edges = [
        (0, 0, 0.),
        (0, 1, 1.),
    ]
    g = networkx.Graph()
    g.add_weighted_edges_from(provided_edges)
    returned_nodes, returned_edges, returned_weights = \
        _parse_networkx_graph(g)
    for (source, target, _) in provided_edges:
        assert (source, target) in returned_edges
    assert {0, 1} == set(returned_nodes)
    assert {(0, 0), (0, 1)} == set(returned_weights.keys())
    assert {0., 1.} == set(returned_weights.values())


def test_igraph_graph():
    import igraph
    # 1) unweighted
    provided_edges = [
        (0, 0),
        (0, 1),
    ]
    g = igraph.Graph(provided_edges)
    returned_nodes, returned_edges, returned_weights = \
        _parse_igraph_graph(g)
    for edge in provided_edges:
        assert edge in returned_edges
    assert {0, 1} == set(returned_nodes)
    assert returned_weights is None

    # 2) weighted
    provided_edges = [
        (0, 0, 0.),
        (0, 1, 1.),
    ]
    g = igraph.Graph.TupleList(provided_edges, weights=True)
    returned_nodes, returned_edges, returned_weights = \
        _parse_igraph_graph(g)
    for (source, target, _) in provided_edges:
        assert (source, target) in returned_edges
    assert {0, 1} == set(returned_nodes)
    assert {(0, 0), (0, 1)} == set(returned_weights.keys())
    assert {0., 1.} == set(returned_weights.values())


def test_parse_graph_tool_graph():
    import graph_tool
    g = graph_tool.Graph()
    v1 = g.add_vertex()
    v2 = g.add_vertex()
    e1 = g.add_edge(v1, v1)
    e2 = g.add_edge(v1, v2)
    returned_nodes, returned_edges, returned_weights = \
        _parse_graph_tool_graph(g)
    desired_edges = [
        (0, 0),
        (0, 1),
    ]
    for edge in desired_edges:
        assert edge in returned_edges
    assert {0, 1} == set(returned_nodes)
    assert returned_weights is None


def test_parse_graph():
    # 1) list-like
    provided_edges = [
        (0, 0),
        (0, 1),
    ]
    returned_nodes, returned_edges, returned_weights = \
        parse_graph(provided_edges)
    for edge in provided_edges:
        assert edge in returned_edges
    assert {0, 1} == set(returned_nodes)
    assert returned_weights is None

    # 2) numpy array
    provided_edges = np.array([
        (0, 0),
        (0, 1),
    ])
    returned_nodes, returned_edges, returned_weights = \
        parse_graph(provided_edges)
    for edge in provided_edges:
        assert tuple(edge) in returned_edges
    assert {0, 1} == set(returned_nodes)
    assert returned_weights is None

    # 3) networkx graph
    import networkx
    provided_edges = [
        (0, 0),
        (0, 1),
    ]
    g = networkx.Graph(provided_edges)
    returned_nodes, returned_edges, returned_weights = \
        parse_graph(g)
    for edge in provided_edges:
        assert edge in returned_edges
    assert {0, 1} == set(returned_nodes)
    assert returned_weights is None

    # 4) igraph graph
    import igraph
    provided_edges = [
        (0, 0),
        (0, 1),
    ]
    g = igraph.Graph(provided_edges)
    returned_nodes, returned_edges, returned_weights = \
        parse_graph(g)
    for edge in provided_edges:
        assert edge in returned_edges
    assert {0, 1} == set(returned_nodes)
    assert returned_weights is None

    # 5) graph_tool graph
    import graph_tool
    g = graph_tool.Graph()
    v1 = g.add_vertex()
    v2 = g.add_vertex()
    e1 = g.add_edge(v1, v1)
    e2 = g.add_edge(v1, v2)
    returned_nodes, returned_edges, returned_weights = \
        parse_graph(g)
    desired_edges = [
        (0, 0),
        (0, 1),
    ]
    for edge in desired_edges:
        assert edge in returned_edges
    assert {0, 1} == set(returned_nodes)
    assert returned_weights is None
