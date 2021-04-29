#!/usr/bin/env python
"""
Run tests.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt

from toy_graphs import unbalanced_tree

from netgraph._main import Graph, draw_edges, draw_nodes, BaseGraph
from netgraph._utils import _get_point_on_a_circle


@pytest.mark.mpl_image_compare
def test_Graph():
    fig, ax = plt.subplots()
    g = Graph([(0, 1)], ax=ax)
    ax.set_aspect('equal')
    return fig


@pytest.mark.mpl_image_compare
def test_draw_edges():
    fig, ax = plt.subplots()
    draw_edges([(0, 1)], {0: (0.1,0.1), 1:(0.9,0.9)}, ax=ax)
    ax.set_aspect('equal')
    return fig


@pytest.mark.mpl_image_compare
def test_draw_curved_edges():
    fig, ax = plt.subplots()
    edge_list = [
        (0, 1),
        (0, 2),
    ]
    node_positions = {
        0 : np.array([0.1, 0.1]),
        1 : np.array([0.49, 0.51]),
        2 : np.array([0.9, 0.9]),
    }

    draw_edges(edge_list, node_positions, curved=True, ax=ax)
    draw_nodes(node_positions, ax=ax)
    ax.set_aspect('equal')
    ax.axis([-0.1, 1.1, -0.1, 1.1])
    return fig


@pytest.mark.mpl_image_compare
def test_draw_selfloops():
    nodes = list(range(17))
    edges = list(zip(nodes[:-1], nodes[1:])) + [(nodes[-1], nodes[0])]
    selfloops = [(node, node) for node in nodes]
    edges = edges + selfloops
    fig, ax = plt.subplots()
    BaseGraph(edges, node_layout='circular', edge_layout='curved', arrows=True)
    return fig


@pytest.mark.mpl_image_compare
def test_draw_curved_directed_edges():
    fig, ax = plt.subplots()
    edge_list = [
        (0, 1),
        (1, 0),
        (0, 2),
    ]
    node_positions = {
        0 : np.array([0.1, 0.1]),
        1 : np.array([0.49, 0.51]),
        2 : np.array([0.9, 0.9]),
    }

    draw_edges(edge_list, node_positions, curved=True, ax=ax)
    draw_nodes(node_positions, ax=ax)
    ax.set_aspect('equal')
    ax.axis([-0.1, 1.1, -0.1, 1.1])
    return fig


@pytest.mark.mpl_image_compare
def test_draw_straight_directed_edges():
    fig, ax = plt.subplots()
    edge_list = [
        (0, 1),
        (1, 0),
        (0, 2),
    ]
    node_positions = {
        0 : np.array([0.1, 0.1]),
        1 : np.array([0.5, 0.1]),
        2 : np.array([0.9, 0.9]),
    }
    Graph(edge_list, node_layout=node_positions, edge_layout='straight')
    ax.axis([-0.1, 1.1, -0.1, 1.1])
    return fig


@pytest.mark.mpl_image_compare
def test_draw_curved_directed_edges_with_labels():
    fig, ax = plt.subplots()
    edge_list = [
        (0, 1),
        (1, 0),
        (0, 2),
    ]
    node_positions = {
        0 : np.array([0.1, 0.1]),
        1 : np.array([0.49, 0.51]),
        2 : np.array([0.9, 0.9]),
    }
    edge_labels = dict(zip(edge_list, 'ABC'))
    Graph(edge_list, node_layout=node_positions, edge_labels=edge_labels, edge_layout='curved')
    ax.axis([-0.1, 1.1, -0.1, 1.1])
    return fig


@pytest.mark.mpl_image_compare
def test_draw_straight_directed_edges_with_labels():
    fig, ax = plt.subplots()
    edge_list = [
        (0, 1),
        (1, 0),
        (0, 2),
    ]
    node_positions = {
        0 : np.array([0.1, 0.1]),
        1 : np.array([0.5, 0.1]),
        2 : np.array([0.9, 0.9]),
    }
    edge_labels = dict(zip(edge_list, 'ABC'))
    Graph(edge_list, node_layout=node_positions, edge_labels=edge_labels, edge_layout='straight')
    ax.axis([-0.1, 1.1, -0.1, 1.1])
    return fig


@pytest.mark.mpl_image_compare
def test_draw_node_labels():
    fig, ax = plt.subplots()
    edge_list = [
        (0, 1),
        (1, 0),
        (0, 2),
    ]
    node_positions = {
        0 : np.array([0.1, 0.1]),
        1 : np.array([0.5, 0.1]),
        2 : np.array([0.9, 0.9]),
    }
    node_labels = {
        0 : 'I',
        1 : 'Lorem ipsum'
    }
    Graph(edge_list, node_layout=node_positions, node_labels=node_labels, node_label_fontdict=dict(size=10))
    ax.axis([-0.1, 1.1, -0.1, 1.1])
    return fig


@pytest.mark.mpl_image_compare
def test_draw_node_labels_with_automatic_resize():
    fig, ax = plt.subplots()
    edge_list = [
        (0, 1),
        (1, 0),
        (0, 2),
    ]
    node_positions = {
        0 : np.array([0.1, 0.1]),
        1 : np.array([0.5, 0.1]),
        2 : np.array([0.9, 0.9]),
    }
    node_labels = {
        0 : 'I',
        1 : 'Lorem ipsum'
    }
    Graph(edge_list, node_layout=node_positions, node_labels=node_labels, node_size=10)
    ax.axis([-0.1, 1.1, -0.1, 1.1])
    return fig


@pytest.mark.mpl_image_compare
def test_draw_node_labels_with_offset():
    fig, ax = plt.subplots()
    edge_list = [
        (0, 1),
        (1, 0),
        (0, 2),
    ]
    node_positions = {
        0 : np.array([0.1, 0.1]),
        1 : np.array([0.5, 0.1]),
        2 : np.array([0.9, 0.9]),
    }
    node_labels = {
        0 : 'I',
        1 : 'Lorem ipsum'
    }
    Graph(edge_list, node_layout=node_positions, node_labels=node_labels, node_size=10,
          node_label_offset=(0.1, -0.1), node_label_fontdict=dict(horizontalalignment='left', verticalalignment='top'))
    ax.axis([-0.1, 1.1, -0.1, 1.1])
    return fig


@pytest.mark.mpl_image_compare
def test_draw_bundled_edges():
    fig, ax = plt.subplots()
    edge_list = [(0, 1), (2, 3)]
    node_positions = {
        0 : np.array([0, 0.25]),
        1 : np.array([1, 0.25]),
        2 : np.array([0, 0.75]),
        3 : np.array([1, 0.75]),
    }
    Graph(edge_list, node_layout=node_positions, edge_layout='bundled')
    ax.axis([-0.1, 1.1, -0.1, 1.1])
    return fig


@pytest.mark.mpl_image_compare
def test_scale_compatibility():
    fig, ax = plt.subplots()
    edge_list = [(0, 1), (2, 3), (4, 5)]
    node_positions = {
        0 : np.array([ 0.0, 0.25]),
        1 : np.array([ 1.0, 0.25]),
        2 : np.array([ 0.0, 0.50]),
        3 : np.array([ 1.0, 0.50]),
        4 : np.array([-1.5, 0.75]),
        5 : np.array([ 2.5, 0.75]),
    }
    Graph(edge_list, node_layout=node_positions, edge_layout='bundled')
    ax.axis([-1.6, 2.6, -0.1, 1.1])
    return fig


@pytest.mark.mpl_image_compare
def test_position_compatibility():
    fig, ax = plt.subplots()
    edge_list = [(0, 1), (2, 3), (4, 5)]
    node_positions = {
        0 : np.array([ 0.0, 0.25]),
        1 : np.array([ 1.0, 0.25]),
        2 : np.array([ 0.0, 0.50]),
        3 : np.array([ 1.0, 0.50]),
        4 : np.array([ 0.0, 2.00]),
        5 : np.array([ 1.0, 2.00]),
    }
    ax.axis([-0.1, 1.1, -0.1, 2.1])
    Graph(edge_list, node_layout=node_positions, edge_layout='bundled')
    return fig


@pytest.mark.mpl_image_compare
def test_angle_compatibility():
    fig, ax = plt.subplots()
    edge_list = [(0, 1), (2, 3), (4, 5)]
    node_positions = {
        0 : np.array([ 0.0, 0.25]),
        1 : np.array([ 1.0, 0.25]),
        2 : np.array([ 0.0, 0.50]),
        3 : np.array([ 1.0, 0.50]),
        4 : np.array([ 0.0, 0.55]),
        5 : np.array([ 1.0, 0.95]),
    }
    Graph(edge_list, node_layout=node_positions, edge_layout='bundled')
    ax.axis([-0.1, 1.1, -0.1, 1.1])
    return fig


@pytest.mark.mpl_image_compare
def test_visibility_compatibility():
    fig, ax = plt.subplots()
    edge_list = [(0, 1), (2, 3), (4, 5)]
    node_positions = {
        0 : np.array([ 0.0, 0.]),
        1 : np.array([ 1.0, 0.]),
        2 : np.array([ 1.0, 1.]),
        3 : np.array([ 2.0, 1.]),
        4 : np.array([ 0.0, -np.sqrt(2)]), # i.e. distance between midpoints from (0, 1) to (2, 3) the same as (0, 1) to (4, 5)
        5 : np.array([ 1.0, -np.sqrt(2)]),
    }
    Graph(edge_list, node_layout=node_positions, edge_layout='bundled')
    ax.axis([-0.1, 2.1, -1.5, 1.1])
    return fig


@pytest.mark.mpl_image_compare
def test_draw_star_graph_with_bundled_edges():
    fig, ax = plt.subplots()
    # star graph
    total_edges = 20
    edge_list = [(ii, total_edges) for ii in range(total_edges)]
    origin = (0.5, 0.5)
    radius = 0.5
    node_positions = {ii : _get_point_on_a_circle(origin, radius, 2*np.pi*np.random.rand()) for ii in range(total_edges)}
    node_positions[total_edges] = origin
    node_positions = {k : np.array(v) for k, v in node_positions.items()}
    Graph(edge_list, node_layout=node_positions, edge_layout='bundled')
    ax.axis([-0.1, 1.1, -0.1, 1.1])
    return fig


@pytest.mark.mpl_image_compare
def test_draw_random_graph_with_bundled_edges():
    fig, ax = plt.subplots()
    edge_list = np.random.randint(0, 10, size=(40, 2))
    edge_list = [(source, target) for source, target in edge_list if source != target]
    edge_list = list(set(edge_list))
    bg = BaseGraph(edge_list, edge_layout='bundled', edge_width=0.5, arrows=True)
    # bg = BaseGraph(edge_list, edge_layout='straight', edge_width=0.5, arrows=True)
    ax.axis([-0.1, 1.1, -0.1, 1.1])
    return fig


@pytest.mark.mpl_image_compare
def test_draw_graph_with_random_layout():
    edge_list = np.random.randint(0, 10, size=(40, 2))
    edge_list = [(source, target) for source, target in edge_list if source != target]
    edge_list = list(set(edge_list))
    fig, ax = plt.subplots()
    bg = BaseGraph(edge_list, node_layout='random')
    ax.axis([-0.1, 1.1, -0.1, 1.1])
    return fig


@pytest.mark.mpl_image_compare
def test_draw_graph_with_sugiyama_layout():
    fig, ax = plt.subplots()
    bg = BaseGraph(unbalanced_tree, node_layout='dot')
    # ax.axis([-0.1, 1.1, -0.1, 1.1])
    return fig


@pytest.mark.mpl_image_compare
def test_draw_graph_with_circular_layout():
    fig, ax = plt.subplots()
    bg = BaseGraph(unbalanced_tree, node_layout='circular')
    # ax.axis([-0.1, 1.1, -0.1, 1.1])
    return fig
