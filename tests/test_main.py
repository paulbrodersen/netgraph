#!/usr/bin/env python
"""
Run tests.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt

from netgraph._main import Graph, draw_edges, draw_nodes


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
    fig, ax = plt.subplots()
    edge_list = [
        (0, 1),
        (1, 1),
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
    edge_labels = dict(zip(edge_list, 'ABC'))
    Graph(edge_list, node_positions=node_positions, curved=False)
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
    Graph(edge_list, node_positions=node_positions, edge_labels=edge_labels, curved=True)
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
    Graph(edge_list, node_positions=node_positions, edge_labels=edge_labels, curved=False)
    ax.axis([-0.1, 1.1, -0.1, 1.1])
    return fig
