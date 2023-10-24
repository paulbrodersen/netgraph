#!/usr/bin/env python
"""
Test _multigraph_classes.py.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt

from netgraph._multigraph_classes import (
    BaseMultiGraph,
    MultiGraph,
)


@pytest.mark.mpl_image_compare
def test_basemultigraph_straight_edge_layout():
    edges = [
        # single uni-directional edge
        (0, 0, "a"),
        # single uni-directional self-loop
        (0, 1, "a"),
        # two uni-directional edges
        (1, 1, "a"),
        (1, 1, "b"),
        # two uni-directional self-lops
        (1, 2, "a"),
        (1, 2, "b"),
        # three
        (2, 2, "a"),
        (2, 2, "b"),
        (2, 2, "c"),
        (2, 0, "a"),
        (2, 0, "b"),
        (2, 0, "c"),
        # single bi-directional edge
        (0, 3, "a"),
        (3, 0, "a"),
        # two bi-directional edges
        (1, 3, "a"),
        (1, 3, "b"),
        (3, 1, "a"),
        (3, 1, "b"),
        # one uni-directional edge with one bi-directional edge
        (2, 3, "a"),
        (2, 3, "b"),
        (3, 2, "a"),
    ]
    node_layout = {
        0 : np.array([0.2, 0.2]),
        1 : np.array([0.8, 0.2]),
        2 : np.array([0.8, 0.8]),
        3 : np.array([0.2, 0.8]),
    }
    fig, ax = plt.subplots()
    BaseMultiGraph(
        edges,
        node_layout=node_layout,
        edge_layout="straight",
        node_labels=True,
        edge_width=0.5,
        arrows=True,
        ax=ax,
    )
    return fig


@pytest.mark.mpl_image_compare
def test_basemultigraph_curved_edge_layout():
    edges = [
        # single uni-directional edge
        (0, 0, "a"),
        # single uni-directional self-loop
        (0, 1, "a"),
        # two uni-directional edges
        (1, 1, "a"),
        (1, 1, "b"),
        # two uni-directional self-lops
        (1, 2, "a"),
        (1, 2, "b"),
        # three
        (2, 2, "a"),
        (2, 2, "b"),
        (2, 2, "c"),
        (2, 0, "a"),
        (2, 0, "b"),
        (2, 0, "c"),
        # single bi-directional edge
        (0, 3, "a"),
        (3, 0, "a"),
        # two bi-directional edges
        (1, 3, "a"),
        (1, 3, "b"),
        (3, 1, "a"),
        (3, 1, "b"),
        # one uni-directional edge with one bi-directional edge
        (2, 3, "a"),
        (2, 3, "b"),
        (3, 2, "a"),
    ]
    node_layout = {
        0 : np.array([0.2, 0.2]),
        1 : np.array([0.8, 0.2]),
        2 : np.array([0.8, 0.8]),
        3 : np.array([0.2, 0.8]),
    }
    fig, ax = plt.subplots()
    BaseMultiGraph(
        edges,
        node_layout=node_layout,
        edge_layout="curved",
        node_labels=True,
        arrows=True,
        ax=ax,
    )
    return fig


@pytest.mark.mpl_image_compare
def test_basemultigraph_arc_edge_layout():
    edges = [
        # single uni-directional edge
        (0, 0, "a"),
        # single uni-directional self-loop
        (0, 1, "a"),
        # two uni-directional edges
        (1, 1, "a"),
        (1, 1, "b"),
        # two uni-directional self-lops
        (1, 2, "a"),
        (1, 2, "b"),
        # three
        (2, 2, "a"),
        (2, 2, "b"),
        (2, 2, "c"),
        (2, 0, "a"),
        (2, 0, "b"),
        (2, 0, "c"),
        # single bi-directional edge
        (0, 3, "a"),
        (3, 0, "a"),
        # two bi-directional edges
        (1, 3, "a"),
        (1, 3, "b"),
        (3, 1, "a"),
        (3, 1, "b"),
        # one uni-directional edge with one bi-directional edge
        (2, 3, "a"),
        (2, 3, "b"),
        (3, 2, "a"),
    ]
    node_layout = {
        0 : np.array([0.2, 0.2]),
        1 : np.array([0.8, 0.2]),
        2 : np.array([0.8, 0.8]),
        3 : np.array([0.2, 0.8]),
    }
    fig, ax = plt.subplots()
    BaseMultiGraph(
        edges,
        node_layout=node_layout,
        edge_layout="arc",
        node_labels=True,
        arrows=True,
        ax=ax,
    )
    return fig


@pytest.mark.mpl_image_compare
def test_weighted_multigraph():
    edges = [
        # single uni-directional edge
        (0, 0, "a", 0.1),
        # single uni-directional self-loop
        (0, 1, "a", 0.2),
        # two uni-directional edges
        (1, 1, "a", 0.3),
        (1, 1, "b", 0.4),
        # two uni-directional self-lops
        (1, 2, "a", 0.5),
        (1, 2, "b", 0.6),
        # three
        (2, 2, "a", 0.7),
        (2, 2, "b", 0.8),
        (2, 2, "c", 0.9),
        (2, 0, "a", 1.0),
        (2, 0, "b", 0.0),
        (2, 0, "c", -0.1),
        # single bi-directional edge
        (0, 3, "a", -0.2),
        (3, 0, "a", -0.3),
        # two bi-directional edges
        (1, 3, "a", -0.4),
        (1, 3, "b", -0.5),
        (3, 1, "a", -0.6),
        (3, 1, "b", -0.7),
        # one uni-directional edge with one bi-directional edge
        (2, 3, "a", -0.8),
        (2, 3, "b", -0.9),
        (3, 2, "a", -1.0),
    ]
    node_layout = {
        0 : np.array([0.2, 0.2]),
        1 : np.array([0.8, 0.2]),
        2 : np.array([0.8, 0.8]),
        3 : np.array([0.2, 0.8]),
    }
    fig, ax = plt.subplots()
    MultiGraph(
        edges,
        node_layout=node_layout,
        edge_layout="straight",
        node_labels=True,
        edge_width=0.5,
        edge_alpha=1.,
        arrows=True,
        ax=ax,
    )
    return fig
