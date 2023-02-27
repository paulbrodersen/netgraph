#!/usr/bin/env python
"""
Test _node_layout.py
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt

from itertools import combinations
from random import choice

from netgraph._main import Graph
from toy_graphs import (
    cycle,
    unbalanced_tree,
    balanced_tree,
    triangle,
    cube,
    star,
    chain,
    single_edge,
)

np.random.seed(1)

@pytest.mark.mpl_image_compare
def test_degenerate_layout():
    # Graph with single node.
    import networkx as nx
    fig, ax = plt.subplots()
    g = nx.Graph()
    g.add_node(1)
    Graph(g, ax=ax)
    return fig


@pytest.mark.mpl_image_compare
def test_circular_layout():
    # Simple cycles and trees can always be drawn without edge crossings.
    fig, axes = plt.subplots(1, 2)
    Graph(cycle, node_layout='circular', node_labels=True, ax=axes[0])
    Graph(unbalanced_tree, node_layout='circular', node_labels=True, ax=axes[1])
    return fig


@pytest.mark.mpl_image_compare
def test_linear_layout():
    # Simple cycles and trees can always be drawn without edge crossings.
    fig, axes = plt.subplots(1, 2)
    Graph(cycle, node_layout='linear', node_labels=True, edge_layout='arc', ax=axes[0])
    Graph(unbalanced_tree, node_layout='linear', node_labels=True, edge_layout='arc', ax=axes[1])
    return fig


@pytest.mark.mpl_image_compare
def test_spring_layout():
    fig, axes = plt.subplots(1, 3)
    Graph(triangle, node_layout='spring', ax=axes[0])
    Graph(cube, node_layout='spring', ax=axes[1])
    Graph(star, node_layout='spring', ax=axes[2])
    return fig


@pytest.mark.mpl_image_compare
def test_dot_layout():
    fig, ax = plt.subplots()
    Graph(unbalanced_tree, node_layout='dot', ax=ax)
    return fig


@pytest.mark.mpl_image_compare
def test_radial_layout():
    fig, ax = plt.subplots()
    Graph(balanced_tree, node_layout='radial', ax=ax)
    return fig


@pytest.mark.mpl_image_compare
def test_community_layout():
    partition_sizes = [10, 20, 30, 40]

    edges = [(0, 1), (0, 5), (0, 32), (0, 36), (0, 48), (0, 73), (0, 74), (0, 97), (1, 9), (2, 3), (2, 5), (2, 15), (2, 56), (2, 79), (2, 88), (3, 8), (3, 26), (3, 29), (3, 71), (3, 82), (3, 99), (4, 8), (4, 12), (4, 76), (5, 8), (5, 9), (5, 33), (5, 55), (5, 76), (5, 87), (5, 92), (5, 93), (6, 7), (6, 8), (6, 14), (6, 26), (6, 33), (6, 53), (6, 93), (6, 98), (7, 24), (7, 26), (7, 32), (7, 58), (7, 64), (7, 74), (7, 75), (7, 85), (8, 31), (8, 41), (8, 58), (8, 62), (8, 63), (8, 70), (8, 86), (9, 73), (10, 13), (10, 14), (10, 20), (10, 21), (10, 28), (10, 39), (10, 43), (10, 55), (10, 86), (11, 13), (11, 17), (11, 19), (11, 21), (11, 23), (11, 26), (11, 27), (11, 29), (11, 32), (11, 41), (11, 82), (11, 99), (12, 13), (12, 17), (12, 21), (12, 23), (12, 26), (12, 28), (12, 60), (12, 89), (13, 14), (13, 19), (13, 21), (13, 24), (13, 44), (13, 60), (13, 62), (13, 97), (14, 24), (14, 28), (14, 56), (14, 62), (14, 85), (14, 96), (14, 99), (15, 25), (16, 20), (16, 41), (16, 50), (17, 25), (17, 48), (17, 69), (17, 70), (17, 75), (17, 77), (17, 89), (17, 92), (18, 25), (18, 79), (19, 22), (19, 23), (19, 26), (19, 33), (19, 45), (19, 90), (20, 24), (20, 26), (20, 37), (20, 77), (21, 22), (21, 73), (21, 77), (21, 85), (22, 51), (22, 54), (22, 55), (22, 56), (22, 61), (22, 64), (22, 99), (23, 24), (23, 25), (23, 42), (23, 52), (23, 62), (23, 78), (23, 98), (24, 91), (24, 93), (24, 99), (25, 61), (25, 72), (25, 85), (26, 28), (26, 68), (26, 88), (26, 98), (27, 28), (27, 29), (27, 54), (27, 60), (27, 67), (27, 80), (28, 33), (28, 36), (28, 57), (28, 97), (29, 52), (29, 54), (29, 55), (29, 87), (29, 91), (30, 32), (30, 39), (30, 44), (30, 49), (30, 68), (30, 74), (30, 81), (30, 89), (31, 35), (31, 38), (31, 39), (31, 40), (31, 43), (31, 52), (31, 57), (31, 58), (31, 70), (31, 88), (32, 34), (32, 37), (32, 40), (32, 42), (32, 49), (32, 52), (32, 54), (32, 55), (32, 56), (32, 59), (32, 70), (32, 85), (32, 87), (33, 34), (33, 36), (33, 42), (33, 45), (33, 47), (33, 49), (33, 50), (33, 57), (33, 66), (33, 85), (34, 38), (34, 40), (34, 45), (34, 48), (34, 51), (34, 54), (34, 57), (34, 58), (34, 59), (34, 64), (34, 73), (34, 79), (34, 85), (34, 90), (34, 96), (35, 37), (35, 40), (35, 49), (35, 52), (35, 54), (35, 58), (35, 95), (36, 45), (36, 46), (36, 47), (36, 55), (36, 58), (37, 38), (37, 45), (37, 55), (37, 75), (37, 84), (37, 88), (37, 96), (38, 47), (38, 48), (38, 52), (38, 56), (38, 57), (38, 59), (38, 67), (38, 69), (38, 78), (38, 89), (39, 41), (39, 48), (39, 51), (39, 54), (39, 60), (39, 84), (39, 98), (39, 99), (40, 46), (40, 47), (40, 55), (40, 56), (40, 77), (40, 87), (40, 94), (40, 95), (41, 42), (41, 45), (41, 46), (41, 47), (41, 50), (41, 52), (41, 56), (41, 98), (42, 51), (42, 57), (42, 59), (42, 61), (42, 63), (43, 45), (43, 46), (43, 47), (43, 55), (43, 59), (43, 67), (44, 49), (44, 53), (44, 56), (45, 47), (45, 53), (45, 55), (46, 53), (46, 57), (46, 60), (46, 73), (46, 97), (47, 52), (47, 57), (47, 58), (47, 92), (47, 98), (48, 49), (49, 67), (49, 96), (50, 54), (50, 55), (50, 58), (50, 75), (51, 53), (51, 54), (51, 56), (51, 59), (51, 83), (53, 56), (53, 71), (53, 80), (55, 77), (56, 58), (56, 65), (56, 99), (57, 60), (57, 77), (58, 70), (58, 82), (58, 86), (59, 73), (59, 76), (59, 96), (60, 61), (60, 65), (60, 68), (60, 69), (60, 86), (60, 95), (60, 96), (61, 64), (61, 68), (61, 69), (61, 73), (61, 77), (61, 78), (61, 86), (61, 88), (61, 89), (61, 91), (61, 93), (61, 94), (61, 99), (62, 68), (62, 70), (62, 72), (62, 75), (62, 83), (62, 87), (62, 88), (62, 92), (62, 94), (62, 95), (62, 98), (63, 64), (63, 65), (63, 68), (63, 74), (63, 82), (63, 85), (63, 91), (63, 92), (63, 93), (63, 96), (64, 65), (64, 69), (64, 72), (64, 76), (64, 78), (64, 80), (64, 84), (64, 86), (64, 88), (64, 89), (64, 90), (64, 91), (64, 94), (64, 98), (65, 74), (65, 76), (65, 77), (65, 82), (65, 83), (65, 85), (65, 89), (66, 67), (66, 69), (66, 74), (66, 77), (66, 78), (66, 82), (66, 83), (66, 84), (66, 85), (66, 90), (66, 92), (66, 93), (66, 98), (67, 69), (67, 74), (67, 76), (67, 80), (67, 88), (67, 89), (67, 90), (67, 91), (67, 92), (67, 94), (67, 95), (67, 97), (68, 69), (68, 75), (68, 82), (68, 83), (68, 84), (68, 92), (68, 95), (68, 96), (68, 99), (69, 74), (69, 82), (69, 83), (69, 87), (69, 90), (69, 91), (69, 92), (69, 93), (69, 96), (69, 99), (70, 73), (70, 80), (70, 84), (70, 88), (70, 89), (70, 93), (70, 96), (70, 97), (71, 81), (71, 95), (72, 77), (72, 78), (72, 79), (72, 81), (72, 82), (72, 96), (72, 98), (73, 75), (73, 79), (73, 83), (73, 85), (73, 88), (73, 89), (73, 93), (73, 96), (74, 88), (74, 92), (75, 76), (75, 86), (75, 87), (75, 88), (75, 89), (75, 91), (75, 96), (76, 81), (76, 85), (76, 87), (77, 78), (77, 80), (77, 82), (77, 85), (77, 91), (77, 93), (77, 96), (78, 88), (78, 92), (78, 93), (78, 95), (79, 82), (79, 83), (79, 88), (79, 89), (79, 93), (79, 98), (79, 99), (80, 83), (80, 99), (81, 83), (81, 91), (81, 92), (81, 98), (81, 99), (82, 85), (82, 87), (82, 90), (82, 95), (83, 89), (83, 95), (84, 86), (84, 91), (84, 95), (84, 97), (84, 98), (85, 88), (85, 89), (85, 91), (85, 93), (85, 94), (86, 90), (86, 94), (86, 96), (87, 90), (87, 94), (87, 95), (89, 91), (89, 96), (90, 99), (91, 93), (91, 94), (91, 97), (93, 95), (93, 96), (94, 95), (94, 97), (94, 98), (95, 96), (95, 98), (95, 99), (96, 98)]

    node_to_community = dict()
    node = 0
    for pid, size in enumerate(partition_sizes):
        for _ in range(size):
            node_to_community[node] = pid
            node += 1

    community_to_color = {
        0 : 'tab:blue',
        1 : 'tab:orange',
        2 : 'tab:green',
        3 : 'tab:red',
    }
    node_color = {node: community_to_color[community] for node, community in node_to_community.items()}

    fig, ax = plt.subplots()
    Graph(edges,
          node_color=node_color, node_edge_width=0, edge_alpha=0.1,
          node_layout='community', node_layout_kwargs=dict(node_to_community=node_to_community),
          # edge_layout='bundled', edge_layout_kwargs=dict(k=2000), # too slow
    )
    return fig


@pytest.mark.mpl_image_compare
def test_community_layout_rotation():
    triangle = [(0, 1), (1, 2), (2, 0)]
    square = [(3, 4), (4, 5), (5, 6), (6, 3)]
    edges = triangle + square + [(0, 3)]

    node_to_community = {
        0 : 0,
        1 : 0,
        2 : 0,
        3 : 1,
        4 : 1,
        5 : 1,
        6 : 1,
    }
    community_to_color = {
        0 : 'tab:blue',
        1 : 'tab:orange',
    }
    node_color = {node: community_to_color[community] for node, community in node_to_community.items()}

    fig, ax = plt.subplots()
    Graph(edges,
          node_color=node_color, node_edge_width=0, edge_alpha=0.1,
          node_layout='community', node_layout_kwargs=dict(node_to_community=node_to_community),
    )
    return fig


@pytest.fixture
def multi_component_graph():
    edges = []

    # add 15 2-node components
    edges.extend([(ii, ii+1) for ii in range(30, 60, 2)])

    # add 10 3-node components
    for ii in range(60, 90, 3):
        edges.extend([(ii, ii+1), (ii, ii+2), (ii+1, ii+2)])

    # add a couple of larger components
    n = 90
    for ii in [10, 20, 30]:
        edges.extend(list(combinations(range(n, n+ii), 2)))
        n += ii

    nodes = list(range(n))
    return nodes, edges


@pytest.mark.mpl_image_compare
def test_circular_layout_with_multiple_components(multi_component_graph):
    nodes, edges = multi_component_graph
    fig, ax = plt.subplots()
    Graph(edges, nodes=nodes, node_size=1, edge_width=0.3, node_layout='circular', ax=ax)
    return fig


@pytest.mark.mpl_image_compare
def test_bipartite_layout():
    fig, axes = plt.subplots(1, 3)
    Graph(single_edge,                             node_layout='bipartite', node_labels=True, ax=axes[0])
    Graph(chain,                                   node_layout='bipartite', node_labels=True, ax=axes[1])
    Graph([(0, 1), (2, 3)], nodes=[0, 1, 2, 3, 4], node_layout='bipartite', node_labels=True, ax=axes[2])
    return fig


@pytest.fixture
def multipartite_graph():
    partitions = [
        list(range(3)),
        list(range(3, 9)),
        list(range(9, 21))
    ]

    edges = list(zip(np.repeat(partitions[0], 2), partitions[1])) \
          + list(zip(np.repeat(partitions[0], 2), partitions[1][1:])) \
          + list(zip(np.repeat(partitions[1], 2), partitions[2])) \
          + list(zip(np.repeat(partitions[1], 2), partitions[2][1:]))

    return partitions, edges


@pytest.mark.mpl_image_compare
def test_multipartite_layout(multipartite_graph):
    layers, edges = multipartite_graph
    fig, ax = plt.subplots()
    Graph(edges, node_layout='multipartite', node_layout_kwargs=dict(layers=layers, reduce_edge_crossings=False), node_labels=True, ax=ax)
    return fig


@pytest.mark.mpl_image_compare
def test_shell_layout(multipartite_graph):
    shells, edges = multipartite_graph
    fig, ax = plt.subplots()
    Graph(edges, node_layout='shell', node_layout_kwargs=dict(shells=shells, reduce_edge_crossings=False), node_labels=True, ax=ax)
    return fig


@pytest.mark.mpl_image_compare
def test_remove_overlap():
    edges = [(0, 1)]
    node_size = {0 : 71, 1 : 71}
    fig, ax = plt.subplots()
    Graph(edges, node_size=node_size, ax=ax)
    return fig


@pytest.mark.mpl_image_compare
def test_geometric_layout():
    fig, (ax1, ax2) = plt.subplots(1, 2)

    right_triangle = {
        (0, 1) : 0.3,
        (1, 2) : 0.4,
        (2, 0) : 0.5,
    }
    Graph(list(right_triangle.keys()), node_layout='geometric', node_layout_kwargs=dict(edge_length=right_triangle), ax=ax1)

    square = {
        (0, 1) : 0.5,
        (1, 2) : 0.5,
        (2, 3) : 0.5,
        (3, 0) : 0.5,
    }
    Graph(list(square.keys()), node_layout='geometric', node_layout_kwargs=dict(edge_length=square), ax=ax2)

    return fig
