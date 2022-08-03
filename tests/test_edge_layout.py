#!/usr/bin/env python
"""
Test _edge_layout.py
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt

from netgraph._main import Graph
from netgraph._utils import _get_point_on_a_circle
from toy_graphs import star

np.random.seed(42)


@pytest.mark.mpl_image_compare
def test_straight_edge_layout():
    edges = [(0, 0), (0, 1), (1, 0), (1, 2), (2, 0)]
    node_positions = {
        0 : (0.2, 0.2),
        1 : (0.5, 0.8),
        2 : (0.8, 0.2),
    }
    fig, ax = plt.subplots()
    Graph(edges, node_layout=node_positions, edge_layout='straight', arrows=True)
    return fig


@pytest.mark.mpl_image_compare
def test_curved_edge_layout():
    fig, ax = plt.subplots()
    edges = [
        (0, 1),
        (1, 0),
        (0, 2),
        (2, 2),
    ]
    node_positions = {
        0 : np.array([0.1, 0.1]),
        1 : np.array([0.5, 0.5]),
        2 : np.array([0.9, 0.89]),
    }
    Graph(edges, node_layout=node_positions, edge_layout='curved')
    return fig


@pytest.mark.mpl_image_compare
def test_arced_edge_layout():
    fig, ax = plt.subplots()
    edges = [
        (0, 1),
    ]
    node_positions = {
        0 : np.array([0.1, 0.5]),
        1 : np.array([0.9, 0.5])
    }
    Graph(edges, node_layout=node_positions, edge_layout='arc', edge_layout_kwargs=dict(rad=1.))
    return fig


# --------------------------------------------------------------------------------
# bundled edge layout

@pytest.mark.mpl_image_compare
def test_draw_bundled_edges():
    fig, ax = plt.subplots()
    edges = [(0, 1), (2, 3)]
    node_positions = {
        0 : np.array([0, 0.25]),
        1 : np.array([1, 0.25]),
        2 : np.array([0, 0.75]),
        3 : np.array([1, 0.75]),
    }
    Graph(edges, node_layout=node_positions, edge_layout='bundled', ax=ax)
    return fig


@pytest.mark.mpl_image_compare
def test_scale_compatibility():
    fig, ax = plt.subplots()
    edges = [(0, 1), (2, 3), (4, 5)]
    node_positions = {
        0 : np.array([ 0.0, 0.25]),
        1 : np.array([ 1.0, 0.25]),
        2 : np.array([ 0.0, 0.50]),
        3 : np.array([ 1.0, 0.50]),
        4 : np.array([-1.5, 0.75]),
        5 : np.array([ 2.5, 0.75]),
    }
    Graph(edges, node_layout=node_positions, edge_layout='bundled', ax=ax)
    ax.axis([-1.6, 2.6, -0.1, 1.1])
    return fig


@pytest.mark.mpl_image_compare
def test_position_compatibility():
    fig, ax = plt.subplots()
    edges = [(0, 1), (2, 3), (4, 5)]
    node_positions = {
        0 : np.array([ 0.0, -1.0]),
        1 : np.array([ 1.0, -1.0]),
        2 : np.array([ 0.0, 0.0]),
        3 : np.array([ 1.0, 0.0]),
        4 : np.array([ 0.0, 4.0]),
        5 : np.array([ 1.0, 4.0]),
    }
    Graph(edges, node_layout=node_positions, edge_layout='bundled', ax=ax)
    ax.axis([-0.1, 1.1, -1.1, 4.1])
    return fig


@pytest.mark.mpl_image_compare
def test_angle_compatibility():
    fig, ax = plt.subplots()
    edges = [(0, 1), (2, 3), (4, 5)]
    node_positions = {
        0 : np.array([ 0.0, 0.25]),
        1 : np.array([ 1.0, 0.25]),
        2 : np.array([ 0.0, 0.50]),
        3 : np.array([ 1.0, 0.50]),
        4 : np.array([ 0.0, 0.55]),
        5 : np.array([ 1.0, 0.95]),
    }
    Graph(edges, node_layout=node_positions, edge_layout='bundled', ax=ax)
    return fig


@pytest.mark.mpl_image_compare
def test_visibility_compatibility():
    fig, ax = plt.subplots()
    edges = [(0, 1), (2, 3), (4, 5)]
    node_positions = {
        0 : np.array([ 0.0, 0.]),
        1 : np.array([ 1.0, 0.]),
        2 : np.array([ 1.0, 1.]),
        3 : np.array([ 2.0, 1.]),
        4 : np.array([ 0.0, -np.sqrt(2)]), # i.e. distance between midpoints from (0, 1) to (2, 3) the same as (0, 1) to (4, 5)
        5 : np.array([ 1.0, -np.sqrt(2)]),
    }
    Graph(edges, node_layout=node_positions, edge_layout='bundled', ax=ax)
    ax.axis([-0.1, 2.1, -1.5, 1.1])
    return fig


@pytest.mark.mpl_image_compare
def test_draw_star_graph_with_bundled_edges():
    fig, ax = plt.subplots()
    total_edges = len(star)
    origin = (0.5, 0.5)
    radius = 0.5
    node_positions = {ii+1 : _get_point_on_a_circle(origin, radius, 2*np.pi*np.random.rand()) for ii in range(total_edges)}
    node_positions[0] = origin
    node_positions = {k : np.array(v) for k, v in node_positions.items()}
    Graph(star, node_layout=node_positions, edge_layout='bundled', ax=ax)
    return fig
