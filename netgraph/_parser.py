#!/usr/bin/env python
# coding: utf-8
"""
Functions for parsing various graph formats and converting them into a node list, edge list, and edge-to-weight mapping.
"""

import warnings
import numpy as np

from functools import wraps

from ._utils import _save_cast_float_to_int, _get_unique_nodes


def _handle_multigraphs(parser):
    """Raise a warning if the given graph appears to be a multigraph, and remove duplicate edges."""
    def wrapped_parser(graph, *args, **kwargs):
        nodes, edges, edge_weight = parser(graph, *args, **kwargs)

        new_edges = list(set([(edge[0], edge[1]) for edge in edges]))
        if len(new_edges) < len(edges):
            msg = "Multi-graphs are not properly supported. Duplicate edges are plotted as a single edge; edge weights (if any) are summed."
            warnings.warn(msg)
            if edge_weight: # sum weights
                new_edge_weight = dict()
                for edge, weight in edge_weight.items():
                    if (edge[0], edge[1]) in new_edge_weight:
                        new_edge_weight[(edge[0], edge[1])] += weight
                    else:
                        new_edge_weight[(edge[0], edge[1])] = weight
            else:
                new_edge_weight = edge_weight
            return nodes, new_edges, new_edge_weight

        return nodes, edges, edge_weight

    return wrapped_parser


def parse_graph(graph):
    """Parse the given graph format and convert it into a node list, edge list, and edge_weight dictionary.

    Parameters
    ----------
    graph: various formats

        Graph object to plot. Various input formats are supported.
        In order of precedence:

        - Edge list:
          Iterable of (source, target) or (source, target, weight) tuples,
          or equivalent (E, 2) or (E, 3) ndarray (where E is the number of edges).
        - Adjacency matrix:
          Full-rank (V, V) ndarray (where V is the number of nodes/vertices).
          The absence of a connection is indicated by a zero.

          .. note:: If V <= 3, any (2, 2) or (3, 3) matrices will be interpreted as edge lists.**

        - networkx.Graph, igraph.Graph, or graph_tool.Graph object

    Returns
    -------
    nodes : list
        List of V unique nodes.
    edges: list of 2-tuples
        List of E edges. Each tuple corresponds to an edge defined by (source node, target node).
    edge_weight: dict edge : float or None
        Dictionary mapping edges to weights. If the graph is unweighted, None is returned.

    """

    if isinstance(graph, (list, tuple, set)):
        return _parse_sparse_matrix_format(graph)

    elif isinstance(graph, np.ndarray):
        rows, columns = graph.shape
        if columns in (2, 3):
            return _parse_sparse_matrix_format(graph)
        elif rows == columns:
            return _parse_adjacency_matrix(graph)
        else:
            msg = "Could not interpret input graph."
            msg += "\nIf a graph is specified as a numpy array, it has to have one of the following shapes:"
            msg += "\n\t-(E, 2) or (E, 3), where E is the number of edges"
            msg += "\n\t-(V, V), where V is the number of nodes (i.e. full rank)"
            msg += f"However, the given graph had shape {graph.shape}."

    # this is a terrible way to test for the type but we don't want to import
    # igraph unless we already know that it is available
    elif str(graph.__class__) == "<class 'igraph.Graph'>":
        return _parse_igraph_graph(graph)

    # ditto
    elif str(graph.__class__) in ("<class 'networkx.classes.graph.Graph'>",
                                  "<class 'networkx.classes.digraph.DiGraph'>",
                                  "<class 'networkx.classes.multigraph.MultiGraph'>",
                                  "<class 'networkx.classes.multidigraph.MultiDiGraph'>"):
        return _parse_networkx_graph(graph)

    elif str(graph.__class__) == "<class 'graph_tool.Graph'>":
        return _parse_graph_tool_graph(graph)

    else:
        allowed = ['list', 'tuple', 'set', 'networkx.Graph', 'igraph.Graph', 'graphtool.Graph']
        raise NotImplementedError("Input graph must be one of: {}\nCurrently, type(graph) = {}".format("\n\n\t" + "\n\t".join(allowed), type(graph)))


@_handle_multigraphs
def _parse_sparse_matrix_format(adjacency):
    """Parse graphs given in a sparse format, i.e. edge lists or sparse matrix representations."""
    adjacency = np.array(adjacency)
    rows, columns = adjacency.shape

    if columns == 2:
        edges = _parse_edge_list(adjacency)
        nodes = _get_unique_nodes(edges)
        return nodes, edges, None

    elif columns == 3:
        edges = _parse_edge_list(adjacency[:, :2])
        nodes = _get_unique_nodes(edges)
        edge_weight = {(source, target) : weight for (source, target, weight) in adjacency}

        # In a sparse adjacency format with integer nodes and float weights,
        # the type of nodes is promoted to the same type as weights.
        # If all nodes can safely be demoted to ints, then we probably want to do that.
        save = True
        for node in nodes:
            if not isinstance(_save_cast_float_to_int(node), int):
                save = False
                break
        if save:
            nodes = [_save_cast_float_to_int(node) for node in nodes]
            edges = [(_save_cast_float_to_int(source), _save_cast_float_to_int(target)) for (source, target) in edges]
            edge_weight = {(_save_cast_float_to_int(source), _save_cast_float_to_int(target)) : weight for (source, target), weight in edge_weight.items()}

        if len(set(edge_weight.values())) > 1:
            return nodes, edges, edge_weight
        else:
            return nodes, edges, None

    else:
        msg = "Graph specification in sparse matrix format needs to consist of an iterable of tuples of length 2 or 3."
        msg += "Got iterable of tuples of length {}.".format(columns)
        raise ValueError(msg)


def _parse_edge_list(edges):
    """Ensures that the type of edges is a list, and each edge is a 2-tuple."""
    # Edge list may be an array, or a list of lists. We want a list of tuples.
    return [(source, target) for (source, target) in edges]


def _parse_adjacency_matrix(adjacency):
    """Parse graphs given in adjacency matrix format, i.e. a full-rank matrix."""
    sources, targets = np.where(adjacency)
    edges = list(zip(sources.tolist(), targets.tolist()))
    nodes = list(range(adjacency.shape[0]))
    edge_weights = {(source, target): adjacency[source, target] for (source, target) in edges}

    if len(set(list(edge_weights.values()))) == 1:
        return nodes, edges, None
    else:
        return nodes, edges, edge_weights


@_handle_multigraphs
def _parse_networkx_graph(graph, attribute_name='weight'):
    """Parse graphs represented as networkx.Graph or related objects."""
    edges = list(graph.edges)
    nodes = list(graph.nodes)
    try:
        edge_weights = {edge : graph.get_edge_data(*edge)[attribute_name] for edge in edges}
    except KeyError: # no weights
        edge_weights = None
    return nodes, edges, edge_weights


@_handle_multigraphs
def _parse_igraph_graph(graph):
    """Parse graphs given as igraph.Graph or related objects."""
    edges = [(edge.source, edge.target) for edge in graph.es()]
    nodes = graph.vs.indices
    if graph.is_weighted():
        edge_weights = {(edge.source, edge.target) : edge['weight'] for edge in graph.es()}
    else:
        edge_weights = None
    return nodes, edges, edge_weights


@_handle_multigraphs
def _parse_graph_tool_graph(graph):
    """Parse graphs given as graph_tool.Graph."""
    nodes = graph.get_vertices().tolist()
    edges = [tuple(edge) for edge in graph.get_edges()]
    # In graph-tool, edge weights are in separate data structure called an edge property map.
    edge_weights = None
    return nodes, edges, edge_weights


def _is_directed(edges):
    """Check if the edge list contains bi-directional edges, i.e. at least one edge (a, b) for which (b, a) also exists."""
    for (source, target) in edges:
        if ((target, source) in edges) and (source != target):
            return True
    return False
