#!/usr/bin/env python
"""
Test _multigraph.py.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt

from netgraph._multigraph import BaseMultiGraph

@pytest.mark.mpl_image_compare
def test_multigraph_straight_edge_layout():
    edges = [
        (0, 1, "a"),
        (0, 1, "b"),
        (0, 1, "c"),
        (0, 0, "d"),
        (0, 0, "e"),
        (0, 0, "f"),
    ]
    node_layout = {
        0 : np.array([0.25, 0.5]),
        1 : np.array([0.75, 0.5]),
    }
    fig, ax = plt.subplots()
    BaseMultiGraph(
        edges,
        node_layout=node_layout,
        edge_layout="straight",
        node_labels=True,
        arrows=True,
        ax=ax,
    )
    return fig


@pytest.mark.mpl_image_compare
def test_multigraph_curved_edge_layout():
    edges = [
        (0, 1, "a"),
        (0, 1, "b"),
        (0, 1, "c"),
        (0, 0, "d"),
        (0, 0, "e"),
        (0, 0, "f"),
    ]
    node_layout = {
        0 : np.array([0.25, 0.5]),
        1 : np.array([0.75, 0.5]),
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
def test_multigraph_arc_edge_layout():
    edges = [
        (0, 1, "a"),
        (0, 1, "b"),
        (0, 1, "c"),
        (0, 0, "d"),
        (0, 0, "e"),
        (0, 0, "f"),
    ]
    node_layout = {
        0 : np.array([0.25, 0.5]),
        1 : np.array([0.75, 0.5]),
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
