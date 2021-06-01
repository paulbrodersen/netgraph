#!/usr/bin/env python
"""
Test _main.py.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt

from netgraph._main import Graph
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
