#!/usr/bin/env python
import numpy as np
import warnings
# # monkey patch warnings
# import colorama
# def custom_formatwarning(msg, *args, **kwargs):
#     # ignore everything except the message
#     return colorama.Fore.RED + "Warning: " + str(msg) + '\n' + colorama.Style.RESET_ALL
# warnings.formatwarning = custom_formatwarning

from scipy.interpolate import BSpline


def _save_cast_float_to_int(num):
    if isinstance(num, float) and np.isclose(num, int(num)):
        return int(num)
    else:
        return num


def _flatten(nested_list):
    return [item for sublist in nested_list for item in sublist]


def _get_unique_nodes(edge_list):
    """
    Using numpy.unique promotes nodes to numpy.float/numpy.int/numpy.str,
    and breaks for nodes that have a more complicated type such as a tuple.
    """
    return list(set(_flatten(edge_list)))


def _edge_list_to_adjacency_matrix(edge_list, edge_weights=None):

    sources = [s for (s, _) in edge_list]
    targets = [t for (_, t) in edge_list]
    if edge_weights:
        weights = [edge_weights[edge] for edge in edge_list]
    else:
        weights = np.ones((len(edge_list)))

    # map nodes to consecutive integers
    nodes = sources + targets
    unique = set(nodes)
    indices = range(len(unique))
    node_to_idx = dict(zip(unique, indices))

    source_indices = [node_to_idx[source] for source in sources]
    target_indices = [node_to_idx[target] for target in targets]

    total_nodes = len(unique)
    adjacency_matrix = np.zeros((total_nodes, total_nodes))
    adjacency_matrix[source_indices, target_indices] = weights

    return adjacency_matrix


def _edge_list_to_adjacency_list(edge_list):
    adjacency = dict()
    for source, target in edge_list:
        if source in adjacency:
            adjacency[source] |= set([target])
        else:
            adjacency[source] = set([target])
    return adjacency


def _get_subgraph(edge_list, node_list):
    subgraph_edge_list = [(source, target) for source, target in edge_list if (source in node_list) and (target in node_list)]
    return subgraph_edge_list


# Adapted from https://stackoverflow.com/a/35007804/2912349
def bspline(cv, n=100, degree=3, periodic=False):
    """ Calculate n samples on a bspline

        cv :      Array ov control vertices
        n  :      Number of samples to return
        degree:   Curve degree
        periodic: True - Curve is closed
    """

    cv = np.asarray(cv)
    count = cv.shape[0]

    # Closed curve
    if periodic:
        kv = np.arange(-degree,count+degree+1)
        factor, fraction = divmod(count+degree+1, count)
        cv = np.roll(np.concatenate((cv,) * factor + (cv[:fraction],)),-1,axis=0)
        degree = np.clip(degree,1,degree)

    # Opened curve
    else:
        degree = np.clip(degree,1,count-1)
        kv = np.clip(np.arange(count+degree+1)-degree,0,count-degree)

    # Return samples
    max_param = count - (degree * (1-periodic))
    spl = BSpline(kv, cv, degree)
    return spl(np.linspace(0,max_param,n))
