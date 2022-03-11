#!/usr/bin/env python
"""
Test _named.py.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt

from netgraph._named import BaseArcDiagram

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
