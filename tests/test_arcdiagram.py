#!/usr/bin/env python
"""
Test _named.py.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt

from netgraph import BaseArcDiagram, ArcDiagram

np.random.seed(42)


@pytest.mark.mpl_image_compare
def test_BaseArcDiagram_defaults():
    fig, ax = plt.subplots()
    BaseArcDiagram([(0, 1)], ax=ax)
    return fig


@pytest.mark.mpl_image_compare
def test_BaseArcDiagram_with_arcs_below():
    fig, ax = plt.subplots()
    BaseArcDiagram([(0, 1)], above=False, ax=ax)
    return fig


@pytest.mark.mpl_image_compare
def test_BaseArcDiagram_with_multiple_components():
    fig, ax = plt.subplots()
    BaseArcDiagram([(0, 1), (2, 3)], nodes=[0, 1, 2, 3, 4], ax=ax)
    return fig


@pytest.mark.mpl_image_compare
def test_BaseArcDiagram_with_custom_node_order():
    fig, ax = plt.subplots()
    BaseArcDiagram([(0, 1), (1, 2)], node_order=[2, 1, 0], node_labels=True, ax=ax)
    return fig


@pytest.mark.mpl_image_compare
def test_BaseArcDiagram_with_custom_node_positions():
    fig, ax = plt.subplots()
    BaseArcDiagram([(0, 1)], node_labels=True, node_layout={0 : (0.1, 0.1), 1 : (0.9, 0.9)}, ax=ax)
    return fig


@pytest.mark.mpl_image_compare
def test_BaseArcDiagram_with_selfloops():
    fig, ax = plt.subplots()
    BaseArcDiagram([(0, 0), (0, 1)], node_labels=True, above=False, ax=ax)
    return fig


@pytest.mark.mpl_image_compare
def test_ArcDiagram_defaults():
    fig, ax = plt.subplots()
    ArcDiagram([(0, 1)], ax=ax)
    return fig


@pytest.mark.mpl_image_compare
def test_ArcDiagram_with_weights():
    fig, ax = plt.subplots()
    ArcDiagram([(0, 1, 1.), (1, 2, -1.), (2, 3, 0.5), (3, 1, -0.5)], ax=ax)
    return fig
