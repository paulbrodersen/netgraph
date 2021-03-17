#!/usr/bin/env python
"""
Functions for reading and writing graphs.
"""
import numpy as np


def parse_graph(graph):
    """
    Arguments
    ----------
    graph: various formats
        Graph object to plot. Various input formats are supported.
        In order of precedence:
            - Edge list:
                Iterable of (source, target) or (source, target, weight) tuples,
                or equivalent (m, 2) or (m, 3) ndarray.
            - Adjacency matrix:
                Full-rank (n,n) ndarray, where n corresponds to the number of nodes.
                The absence of a connection is indicated by a zero.
            - igraph.Graph object
            - networkx.Graph object

    Returns:
    --------
    edge_list: m-long list of 2-tuples
        List of edges. Each tuple corresponds to an edge defined by (source, target).

    edge_weights: dict (source, target) : float or None
        Edge weights. If the graph is unweighted, None is returned.

    is_directed: bool
        True, if the graph appears to be directed due to
            - the graph object class being passed in (e.g. a networkx.DiGraph), or
            - the existence of bi-directional edges.
    """

    if isinstance(graph, (list, tuple, set)):
        return _parse_sparse_matrix_format(graph)

    elif isinstance(graph, np.ndarray):
        rows, columns = graph.shape
        if columns in (2, 3):
            return _parse_sparse_matrix_format(graph)
        else:
            return _parse_adjacency_matrix(graph)

    # this is terribly unsafe but we don't want to import igraph
    # unless we already know that we need it
    elif str(graph.__class__) == "<class 'igraph.Graph'>":
        return _parse_igraph_graph(graph)

    # ditto
    elif str(graph.__class__) in ("<class 'networkx.classes.graph.Graph'>",
                                  "<class 'networkx.classes.digraph.DiGraph'>",
                                  "<class 'networkx.classes.multigraph.MultiGraph'>",
                                  "<class 'networkx.classes.multidigraph.MultiDiGraph'>"):
        return _parse_networkx_graph(graph)

    else:
        allowed = ['list', 'tuple', 'set', 'networkx.Graph', 'igraph.Graph']
        raise NotImplementedError("Input graph must be one of: {}\nCurrently, type(graph) = {}".format("\n\n\t" + "\n\t".join(allowed)), type(graph))


def _parse_edge_list(edge_list):
    # Edge list may be an array, or a list of lists.
    # We want a list of tuples.
    return [(source, target) for (source, target) in edge_list]


def _parse_sparse_matrix_format(adjacency):
    adjacency = np.array(adjacency)
    rows, columns = adjacency.shape
    if columns == 2:
        edge_list = _parse_edge_list(adjacency)
        return edge_list, None, _is_directed(edge_list)
    elif columns == 3:
        edge_list = _parse_edge_list(adjacency[:,:2])
        edge_weights = {(source, target) : weight for (source, target, weight) in adjacency}

        # In a sparse adjacency format with weights,
        # the type of nodes is promoted to the same type as weights,
        # which is commonly a float. If all nodes can safely be demoted to ints,
        # then we probably want to do that.
        tmp = [(_save_cast_float_to_int(source), _save_cast_float_to_int(target)) for (source, target) in edge_list]
        if np.all([isinstance(num, int) for num in _flatten(tmp)]):
            edge_list = tmp

        if len(set(edge_weights.values())) > 1:
            return edge_list, edge_weights, _is_directed(edge_list)
        else:
            return edge_list, None, _is_directed(edge_list)
    else:
        raise ValueError("Graph specification in sparse matrix format needs to consist of an iterable of tuples of length 2 or 3. Got iterable of tuples of length {}.".format(columns))


def _parse_adjacency_matrix(adjacency):
    sources, targets = np.where(adjacency)
    edge_list = list(zip(sources.tolist(), targets.tolist()))
    edge_weights = {(source, target): adjacency[source, target] for (source, target) in edge_list}
    if len(set(list(edge_weights.values()))) == 1:
        return edge_list, None, _is_directed(edge_list)
    else:
        return edge_list, edge_weights, _is_directed(edge_list)


def _parse_networkx_graph(graph, attribute_name='weight'):
    edge_list = list(graph.edges())
    try:
        edge_weights = {edge : graph.get_edge_data(*edge)[attribute_name] for edge in edge_list}
    except KeyError: # no weights
        edge_weights = None
    return edge_list, edge_weights, graph.is_directed()


def _parse_igraph_graph(graph):
    edge_list = [(edge.source, edge.target) for edge in graph.es()]
    if graph.is_weighted():
        edge_weights = {(edge.source, edge.target) : edge['weight'] for edge in graph.es()}
    else:
        edge_weights = None
    return edge_list, edge_weights, graph.is_directed()


def _is_directed(edge_list):
    # test for bi-directional edges
    for (source, target) in edge_list:
        if ((target, source) in edge_list) and (source != target):
            return True
    return False
