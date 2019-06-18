#!/usr/bin/env python
"""
TODO:
- suppress warnings for divide by zero on diagonal -- masked arrays?
- ensure that the adjacency matrix has the correct dimensions even is one of the nodes is unconnected
"""

import numpy as np


def _spring(adjacency, node_positions, origin, scale, temperature):
    """
    Loosely based on Eades (1984), using
    - attraction that scales linearly with distance instead of logarithmically, and
    - repulsion that scales inversely proportional to the square of the distance.
from ._utils import (
    warnings,
    _edge_list_to_adjacency_matrix,
    _edge_list_to_adjacency_list,
    _get_subgraph,
    _get_unique_nodes,
    _flatten,
)

    """

    X = np.array(node_positions.values())

    delta        = X[None, :, ...] - X[:, None, ...]
    distance     = np.sqrt(np.sum(delta*delta, axis=-1))
    direction    = delta / distance[..., None]
    repulsion    = _get_spring_repulsion(distance, direction)
    attraction   = _get_spring_attraction(distance, direction, adjacency)
    displacement = attraction + repulsion

    displacement /= np.max(np.abs(displacement), axis=0) # normalize values to (-1, 1)
    displacement *= scale[None, ...] # rescale to shape of canvas
    displacement *= temperature # scale by temperature

    new_X = X + displacement
    new_X = _rescale_to_frame(new_X, origin, scale)
    # new_X = _enforce_frame(new_X, origin, scale)

    new_node_positions = dict(zip(node_positions.keys(),  new_X))

    return new_node_positions


def _get_spring_repulsion(distance, direction):
    # magnitude = 1. / distance**0.5
    magnitude = 1. / distance**2 # analogous to electrostatical repulsion
    vectors   = direction * magnitude[..., None]
    vectors   = _set_diagonal(vectors, 0)
    return np.sum(vectors, axis=0)


def _get_spring_attraction(distance, direction, adjacency, eps=0.01):
    magnitude = distance * adjacency # analogous to a linear spring
    vectors   = -direction * magnitude[..., None] # NB: the minus!
    vectors   = _set_diagonal(vectors, 0)
    return np.sum(vectors, axis=0)


def _enforce_frame(X, origin, scale):
    minima = origin
    maxima = origin + scale
    for ii, (minimum, maximum) in enumerate(zip(minima, maxima)):
        X[:, ii] = np.clip(X[:, ii], minimum, maximum)
    return X


def _rescale_to_frame(X, origin, scale):
    X -= np.min(X, axis=0)
    X /= np.max(X, axis=0)
    X *= scale[None, ...]
    X += origin[None, ...]
    return X


def _set_diagonal(square_matrix, value=0):
    n = len(square_matrix)
    is_diagonal = np.diag(np.ones((n), dtype=np.bool))
    square_matrix[is_diagonal] = value
    return square_matrix


def _fruchterman_reingold(adjacency, node_positions, origin, scale, temperature, k):

    X = np.array(list(node_positions.values()))

    delta        = X[None, :, ...] - X[:, None, ...]
    distance     = np.linalg.norm(delta, axis=-1)
    direction    = delta / distance[..., None] # i.e. the unit vector
    repulsion    = _get_fr_repulsion(distance, direction, k)
    attraction   = _get_fr_attraction(distance, direction, adjacency, k)
    displacement = attraction + repulsion

    # limit maximum displacement using temperature
    length = np.linalg.norm(displacement, axis=-1)
    new_X = X + displacement / length[:, None] * np.clip(length, None, temperature)[:, None]

    new_X = _enforce_frame(new_X, origin, scale)

    return dict(zip(node_positions.keys(),  new_X))


def _get_fr_repulsion(distance, direction, k):
    magnitude = k**2 / distance
    vectors   = direction * magnitude[..., None]
    # Note that we cannot apply the usual strategy of summing the array
    # along either axis and subtracting the trace,
    # as the diagonal of `direction` is np.nan, and any sum or difference of
    # NaNs is just another NaN.
    # Also we do not want to ignore NaNs by using np.nansum, as then we would
    # potentially mask the existence of off-diagonal zero distances.
    vectors   = _set_diagonal(vectors, 0)
    return np.sum(vectors, axis=0)


def _get_fr_attraction(distance, direction, adjacency, k):
    magnitude = 1/k * distance**2 * adjacency
    vectors   = -direction * magnitude[..., None] # NB: the minus!
    vectors   = _set_diagonal(vectors, 0)
    return np.sum(vectors, axis=0)


def test():

    import matplotlib.pyplot as plt; plt.ion()
    import networkx as nx
    from _main import InteractiveGraph

    # # sparse random graph
    # from _main import _get_random_weight_matrix
    # total_nodes = 100
    # adjacency_matrix = _get_random_weight_matrix(total_nodes, p=0.1,
    #                                              directed=False,
    #                                              strictly_positive=True,
    #                                              weighted=True)

    # sources, targets = np.where(adjacency_matrix)
    # weights = adjacency_matrix[sources, targets]
    # edge_list = np.c_[sources, targets]
    # adjacency = _edge_list_to_adjacency(edge_list)

    # # K-graph
    # total_nodes = 8
    # adjacency = np.ones((total_nodes, total_nodes)) - np.diag(np.ones((total_nodes)))
    # edge_list = nx.from_numpy_matrix(adjacency).edges()

    # # square with satellites
    # edge_list = [
    #     (0, 1),
    #     (1, 2),
    #     (2, 3),
    #     (3, 0),
    #     (0, 4),
    #     (1, 5),
    #     (2, 6),
    #     (3, 7)
    # ]

    # # cycle
    # edge_list = [
    #     (0, 1),
    #     (1, 2),
    #     (2, 3),
    #     (3, 4),
    #     (4, 5),
    #     (5, 6),
    #     (6, 7),
    #     (7, 0)
    # ]

    # # cube
    # edge_list = [
    #     (0, 1),
    #     (1, 2),
    #     (2, 3),
    #     (3, 0),
    #     (4, 5),
    #     (5, 6),
    #     (6, 7),
    #     (7, 4),
    #     (0, 4),
    #     (1, 5),
    #     (2, 6),
    #     (3, 7)
    # ]

    # # star
    # edge_list = [
    #     (0, 1),
    #     (0, 2),
    #     (0, 3),
    #     (0, 4),
    #     (0, 5),
    # ]

    # # unbalanced tree
    # edge_list = [
    #     (0, 1),
    #     (0, 2),
    #     (0, 3),
    #     (0, 4),
    #     (0, 5),
    #     (2, 6),
    #     (3, 7),
    #     (3, 8),
    #     (4, 9),
    #     (4, 10),
    #     (4, 11),
    #     (5, 12),
    #     (5, 13),
    #     (5, 14),
    #     (5, 15)
    # ]

    # adjacency = _edge_list_to_adjacency(edge_list)
    # total_nodes = np.max(edge_list)+1

    node_positions = {ii : np.random.rand(2) for ii in range(total_nodes)}

    fig, ax = plt.subplots(1,1)
    g = InteractiveGraph(edge_list,
                         node_positions=node_positions,
                         ax=ax)

    total_iterations = 50
    x = np.linspace(0, 1, total_iterations) + 1e-4
    temperatures = 0.1 * (x - 1)**2

    for ii, temperature in enumerate(temperatures):
        # g.node_positions = _spring(adjacency, g.node_positions,
        #                            origin=np.zeros((2)),
        #                            scale=np.array((1, 1)),
        #                            temperature=temperature,
        # )
        g.node_positions = _fruchterman_reingold(adjacency, g.node_positions,
                                                 origin=np.zeros((2)),
                                                 scale=np.array((1, 1)),
                                                 temperature=temperature,
                                                 k=np.sqrt(1./total_nodes)
                                                 # k = 0.1,
        )
        g._update_nodes(g.node_positions.keys())
        g._update_edges(g.node_positions.keys())
        g._update_view()
        fig.canvas.draw()

    input("Press any key to close figure...")

if __name__ == '__main__':
    g = test()
