#!/usr/bin/env python
# coding: utf-8
"""
Functions for parsing various graph formats and converting them into a node list, edge list, and edge-to-weight mapping.
"""

import warnings
import numpy as np

from functools import wraps

from ._utils import (
    _get_unique_nodes,
    _save_cast_float_to_int,
)


def _handle_multigraphs(parser):
    """Raise a warning if the given graph appears to be a multigraph, and remove duplicate edges."""
    @wraps(parser)
    def wrapped_parser(graph, *args, **kwargs):
        nodes, edges, edge_weight = parser(graph, *args, **kwargs)

        unique_edges = list(set(edges))
        if len(unique_edges) < len(edges):
            msg = "The given graph data structure appears to be a multi-graph,"
            msg += " however, this netgraph class does not properly support multi-graphs."
            msg += " Instead, the multi-graph is converted into a weighted graph,"
            msg += " in which duplicate edges are merged into one."
            msg += " The corresponding edge weight is set to the number of duplicate edges;"
            msg += " existing edge weights are discarded."
            msg += " Use the `MultiGraph` class and subclasses to visualize all edges and their edge weights properly."
            warnings.warn(msg)

            edge_weight = dict()
            for edge in edges:
                edge_weight[edge] = edge_weight.get(edge, 0) + 1

            return nodes, unique_edges, edge_weight

        return nodes, edges, edge_weight

    return wrapped_parser


def _parse_edge_list(edges):
    """Ensures that the type of edges is a list, and each edge is a 2-tuple."""
    # Edge list may be an array, or a list of lists. We want a list of tuples.
    return [(source, target) for (source, target) in edges]


@_handle_multigraphs
def _parse_sparse_matrix_format(adjacency):
    """Parse graphs given in a sparse format, i.e. edge lists or sparse matrix representations."""
    rows, columns = np.array(adjacency).shape

    if columns == 2:
        edges = _parse_edge_list(adjacency)
        nodes = _get_unique_nodes(edges)
        return nodes, edges, None

    elif columns == 3:
        edge_weight = {(source, target) : weight for (source, target, weight) in adjacency}
        edges = list(edge_weight.keys())
        nodes = _get_unique_nodes(edges)

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


def _parse_nparray(graph):
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
        msg += f"\nHowever, the given graph had shape {graph.shape}."
        raise ValueError(msg)


@_handle_multigraphs
def _parse_networkx_graph(graph, weight_attribute="weight"):
    """Parse graphs represented as networkx.Graph or related objects."""
    edges = list(graph.edges)
    nodes = list(graph.nodes)
    try:
        edge_weights = {edge : graph.get_edge_data(*edge)[weight_attribute] for edge in edges}
    except KeyError: # no weights
        edge_weights = None
    return nodes, edges, edge_weights


@_handle_multigraphs
def _parse_igraph_graph(graph, weight_attribute="weight"):
    """Parse graphs given as igraph.Graph or related objects."""
    edges = [(edge.source, edge.target) for edge in graph.es()]
    nodes = graph.vs.indices
    if graph.is_weighted():
        edge_weights = {(edge.source, edge.target) : edge[weight_attribute] for edge in graph.es()}
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


def _is_listlike(graph):
    return isinstance(graph, (list, tuple, set))

def _is_nparray(graph):
    return isinstance(graph, np.ndarray)

def _is_networkx(graph):
    import networkx
    return isinstance(graph, networkx.Graph)

def _is_igraph(graph):
    import igraph
    return isinstance(graph, igraph.Graph)

def _is_graph_tool(graph):
    import graph_tool
    return isinstance(graph, graph_tool.Graph)


_check_to_parser = {
    _is_listlike   : _parse_sparse_matrix_format,
    _is_nparray    : _parse_nparray,
    _is_networkx   : _parse_networkx_graph,
    _is_igraph     : _parse_igraph_graph,
    _is_graph_tool : _parse_graph_tool_graph,
}


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
    for check, parser in _check_to_parser.items():
        try:
            if check(graph):
                return parser(graph)
        except ModuleNotFoundError:
            pass
    else:
        allowed = ['list', 'tuple', 'set', 'networkx.Graph', 'igraph.Graph', 'graph_tool.Graph']
        raise NotImplementedError("Input graph must be one of: {}\nCurrently, type(graph) = {}".format("\n\n\t" + "\n\t".join(allowed), type(graph)))


# --------------------------------------------------------------------------------


def is_order_zero(graph):
    """Determine if a graph is an order zero graph, i.e. a graph with no nodes (and no edges)."""
    for check, parser in _check_to_parser.items():
        try:
            if check(graph):
                nodes, edges, _ = parser(graph)
                if (not nodes) and (not edges):
                    return True
                else:
                    return False
        except ModuleNotFoundError:
            pass
    else:
        allowed = ['list', 'tuple', 'set', 'networkx.Graph', 'igraph.Graph', 'graph_tool.Graph']
        raise NotImplementedError("Input graph must be one of: {}\nCurrently, type(graph) = {}".format("\n\n\t" + "\n\t".join(allowed), type(graph)))


def is_empty(graph):
    """Determine if a graph is an empty graph, i.e. a graph with nodes but no edges."""
    for check, parser in _check_to_parser.items():
        try:
            if check(graph):
                nodes, edges, _ = parser(graph)
                if nodes and (not edges):
                    return True
                else:
                    return False
        except ModuleNotFoundError:
            pass
    else:
        allowed = ['list', 'tuple', 'set', 'networkx.Graph', 'igraph.Graph', 'graph_tool.Graph']
        raise NotImplementedError("Input graph must be one of: {}\nCurrently, type(graph) = {}".format("\n\n\t" + "\n\t".join(allowed), type(graph)))


# --------------------------------------------------------------------------------
# multi-graph parsers

def _parse_multigraph_edge_list(edges):
    """Ensures that the type of edges is a list, and each edge is a 2-tuple."""
    # Edge list may be an array, or a list of lists. We want a list of tuples.
    return [(source, target, eid) for (source, target, eid) in edges]


def _parse_multigraph_sparse_matrix_format(adjacency):
    """Parse graphs given in a sparse format, i.e. edge lists or sparse matrix representations."""
    rows, columns = np.array(adjacency).shape

    if columns == 3:
        edges = _parse_multigraph_edge_list(adjacency)
        nodes = _get_unique_nodes(edges)
        return nodes, edges, None

    elif columns == 4:
        edge_weight = {(source, target, eid) : weight for (source, target, eid, weight) in adjacency}
        edges = list(edge_weight.keys())
        nodes = _get_unique_nodes(edges)

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
            edges = [(_save_cast_float_to_int(source), _save_cast_float_to_int(target), eid) for (source, target, eid) in edges]
            edge_weight = {(_save_cast_float_to_int(source), _save_cast_float_to_int(target), eid) : weight for (source, target, eid), weight in edge_weight.items()}

        if len(set(edge_weight.values())) > 1:
            return nodes, edges, edge_weight
        else:
            return nodes, edges, None

    else:
        msg = "Multi-graph specification in sparse matrix format needs to consist of an iterable of tuples of length 3 or 4."
        msg += "Got iterable of tuples of length {}.".format(columns)
        raise ValueError(msg)


def _parse_multigraph_adjacency_matrix(adjacency):
    """Parse multi-graphs given in an adjacency matrix format."""
    sources, targets, edge_ids = np.where(adjacency)
    edges = list(zip(sources.tolist(), targets.tolist(), edge_ids.tolist()))
    nodes = list(range(adjacency.shape[0]))
    edge_weights = {(source, target, eid): adjacency[source, target, eid] for (source, target, eid) in edges}

    if len(set(edge_weights.values())) == 1:
        return nodes, edges, None
    else:
        return nodes, edges, edge_weights


def _parse_multigraph_nparray(graph):
    if np.ndim(graph) == 2:
        rows, columns = graph.shape
        if columns in (3, 4):
            return _parse_multigraph_sparse_matrix_format(graph)
    elif np.ndim(graph) == 3:
        rows, columns, layers = graph.shape
        if rows == columns:
            return _parse_multigraph_adjacency_matrix(graph)

    msg = "Could not interpret input graph."
    msg += "\nIf a graph is specified as a numpy array, it has to have one of the following shapes:"
    msg += "\n\t-(E, 3) or (E, 4), where E is the number of edges"
    msg += "\n\t-(V, V, L), where V is the number of nodes and L is the number of layers (i.e. each layer is a full rank matrix)"
    msg += f"\nHowever, the given graph had shape {graph.shape}."
    raise ValueError(msg)


# def _parse_multigraph_networkx_graph(graph, weight_attribute="weight"):
#     """Parse graphs represented as networkx.Graph or related objects."""
#     edges = list(graph.edges)
#     nodes = list(graph.nodes)
#     try:
#         edge_weights = {edge : graph.get_edge_data(*edge)[weight_attribute] \
#                         for edge in edges}
#     except KeyError: # no weights
#         edge_weights = None
#     return nodes, edges, edge_weights
_parse_multigraph_networkx_graph = _parse_networkx_graph.__wrapped__


def _parse_multigraph_igraph_graph(graph, weight_attribute="weight", id_attribute="id"):
    """Parse graphs given as igraph.Graph or related objects."""
    edges = [(edge.source, edge.target, edge[id_attribute]) for edge in graph.es()]
    nodes = graph.vs.indices
    if graph.is_weighted():
        edge_weights = {(edge.source, edge.target, edge[id_attribute]) : edge[weight_attribute] for edge in graph.es()}
    else:
        edge_weights = None
    return nodes, edges, edge_weights


def _parse_multigraph_graph_tool_graph(graph):
    return NotImplementedError("Multi-graph plotting is currently not supported for graph-tool Graph objects.")


def _is_networkx_multigraph(graph):
    import networkx
    return isinstance(graph, networkx.MultiGraph)


_check_to_multigraph_parser = {
    _is_listlike            : _parse_multigraph_sparse_matrix_format,
    _is_nparray             : _parse_multigraph_nparray,
    _is_networkx_multigraph : _parse_multigraph_networkx_graph,
    _is_igraph              : _parse_multigraph_igraph_graph,
    _is_graph_tool          : _parse_multigraph_graph_tool_graph,
}


def parse_multigraph(graph):
    """Parse the given multi-graph format and convert it into a node list, edge list, and edge_weight dictionary.

    Parameters
    ----------
    graph: various formats
        Graph object to plot. Various input formats are supported.
        In order of precedence:

        - Edge list:
          Iterable of (source node ID, target node ID, edge key) or
          (source node ID, target node ID, edge key, weight) tuples,
          or equivalent (E, 3) or (E, 4) ndarray (where E is the number of edges).
        - Adjacency matrix:
          A (V, V, L) ndarray (where V is the number of nodes/vertices, and L is the number of layers).
          The absence of a connection is indicated by a zero.
        - networkx.MultiGraph or igraph.Graph object

    Returns
    -------
    nodes : list
        List of V unique nodes.
    edges: list of 2-tuples
        List of E edges. Each tuple corresponds to an edge defined by (source node, target node).
    edge_weight: dict edge : float or None
        Dictionary mapping edges to weights. If the graph is unweighted, None is returned.

    """
    for check, parser in _check_to_multigraph_parser.items():
        try:
            if check(graph):
                return parser(graph)
        except ModuleNotFoundError:
            pass
    else:
        allowed = ['list', 'tuple', 'set', 'networkx.MultiGraph', 'igraph.Graph']
        raise NotImplementedError("Input graph must be one of: {}\nCurrently, type(graph) = {}".format("\n\n\t" + "\n\t".join(allowed), type(graph)))
