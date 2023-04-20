#!/usr/bin/env python
"""
Test _main.py.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt

from netgraph._main import Graph
from netgraph._artists import Path
from toy_graphs import cube, cycle

np.random.seed(42)


@pytest.fixture
def weighted_cube():
    reverse = [cube[ii][::-1] for ii in np.random.randint(0, 12, size=4)]
    edges = cube + reverse
    weights = np.random.rand(len(edges))-0.5
    return [(source, target, weight) for (source, target), weight in zip(edges, weights)]


@pytest.mark.mpl_image_compare
def test_defaults(weighted_cube):
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    g = Graph(cube, ax=axes[0])
    _ = Graph(weighted_cube, node_layout=g.node_positions, ax=axes[1])
    return fig


@pytest.mark.mpl_image_compare
def test_arrows(weighted_cube):
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    g = Graph(cube, arrows=True, ax=axes[0])
    _ = Graph(weighted_cube, node_layout=g.node_positions, arrows=True, ax=axes[1])
    return fig


@pytest.mark.mpl_image_compare
def test_labels():
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)

    triangle = [(0, 1), (1, 1), (1, 2), (2, 0), (0, 2)]

    node_positions = {
        0 : np.array([0.2, 0.2]),
        1 : np.array([0.5, 0.8]),
        2 : np.array([0.8, 0.2]),
    }

    Graph(triangle, node_layout=node_positions, edge_layout='straight',
          node_labels=True, edge_labels=True, edge_label_position=0.33,
          edge_label_fontdict=dict(fontweight='bold'),
          ax=axes[0])

    Graph(triangle, node_layout=node_positions, edge_layout='curved',
          node_labels={0 : 'Lorem', 2 : 'ipsum'}, node_label_offset=(0.025, 0.025),
          node_label_fontdict=dict(size=15, horizontalalignment='left', verticalalignment='bottom'),
          edge_labels={(1, 2) : 'dolor sit'},
          ax=axes[1])

    return fig


@pytest.mark.mpl_image_compare
def test_update_view():
    fig, ax = plt.subplots()
    edges = [(0, 1)]
    node_layout = {
        0 : np.array([-1, -1]),
        1 : np.array([0.5, 0.5])
    }
    Graph(edges, node_layout=node_layout)
    return fig


@pytest.mark.mpl_image_compare
def test_get_node_label_offset():
    fig, ax = plt.subplots()
    Graph(cycle, node_layout='circular', node_labels=True, node_label_offset=0.1)
    return fig


@pytest.mark.mpl_image_compare
def test_regular_node_shapes():
    node_shape_options = 'so^>v<dph8'
    fig, ax = plt.subplots()
    Graph(cycle, node_layout='circular', node_shape=dict(list(enumerate(node_shape_options))), ax=ax)
    return fig


@pytest.mark.mpl_image_compare
def test_path_node_shapes():
    vertices = np.array([[ 0.44833333, -2.75444444],
                         [-0.78166667, -1.28444444],
                         [-2.88166667,  1.81555556],
                         [-0.75666667,  1.81555556],
                         [-0.28166667,  0.96555556],
                         [ 1.06833333,  3.01555556],
                         [ 1.86833333, -0.13444444],
                         [ 0.86833333, -0.68444444],
                         [ 0.44833333, -2.75444444]])
    codes = (1, 4, 4, 4, 2, 4, 4, 4, 79)
    node_shape = Path(vertices, codes)
    fig, ax = plt.subplots()
    Graph(cycle, node_layout='circular', node_shape=node_shape, ax=ax)
    return fig


@pytest.mark.mpl_image_compare
def test_mixed_node_shapes():
    vertices = np.array([[ 0.44833333, -2.75444444],
                         [-0.78166667, -1.28444444],
                         [-2.88166667,  1.81555556],
                         [-0.75666667,  1.81555556],
                         [-0.28166667,  0.96555556],
                         [ 1.06833333,  3.01555556],
                         [ 1.86833333, -0.13444444],
                         [ 0.86833333, -0.68444444],
                         [ 0.44833333, -2.75444444]])
    codes = (1, 4, 4, 4, 2, 4, 4, 4, 79)
    regular_node_shape_options = 'so^>v<dph8'
    node_shape = dict(list(enumerate(regular_node_shape_options)))
    node_shape[0] = Path(vertices, codes)
    fig, ax = plt.subplots()
    Graph(cycle, node_layout='circular', node_shape=node_shape, node_alpha=0.5, arrows=True, ax=ax)
    return fig


@pytest.mark.mpl_image_compare
def test_offset_straight_edge():
    fig, ax = plt.subplots()
    edges = [(0, 1)]
    node_layout = {
        0 : np.array([0.1, 0.5]),
        1 : np.array([0.9, 0.5])
    }
    node_size = 10
    Graph(edges, node_layout=node_layout, edge_layout='straight', node_shape='^', node_alpha=0.5, node_size=node_size, arrows=True, ax=ax)
    ax.add_artist(plt.Circle(node_layout[1], node_size/100, zorder=-1, color='lightgray'))
    ax.axis([0, 1, 0, 1])
    return fig


@pytest.mark.mpl_image_compare
def test_offset_curved_edge():
    fig, ax = plt.subplots()
    edges = [(0, 1)]
    node_layout = {
        0 : np.array([0.1, 0.5]),
        1 : np.array([0.9, 0.5])
    }
    node_size = 10
    Graph(edges, node_layout=node_layout, edge_layout='curved', node_shape='^', node_alpha=0.5, node_size=node_size, arrows=True, ax=ax)
    ax.add_artist(plt.Circle(node_layout[1], node_size/100, zorder=-1, color='lightgray'))
    ax.axis([0, 1, 0, 1])
    return fig


@pytest.mark.mpl_image_compare
def test_offset_selfloop():
    fig, ax = plt.subplots()
    edges = [(0, 0)]
    node_layout = {
        0 : np.array([0.5, 0.5]),
    }
    node_size = 10
    Graph(edges, node_layout=node_layout, edge_layout='straight', node_shape='^', node_alpha=0.5, node_size=node_size, arrows=True, ax=ax)
    ax.axis([0, 1, 0, 1])
    return fig


@pytest.mark.mpl_image_compare
def test_variable_selfloop_radii():
    fig, ax = plt.subplots()
    edges = [
        (0, 1),
        (1, 2),
        (2, 0),
        (0, 0),
        (1, 1),
        (2, 2),
    ]
    selfloop_radius = {
        (0, 0) : 0.1,
        (1, 1) : 0.2,
        (2, 2) : 0.3,
    }
    Graph(edges, edge_layout_kwargs=dict(selfloop_radius=selfloop_radius), ax=ax)
    return fig
