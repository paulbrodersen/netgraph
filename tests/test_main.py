#!/usr/bin/env python
"""
Run tests.
"""

import pytest
import matplotlib.pyplot as plt

from netgraph._main import Graph

@pytest.mark.mpl_image_compare
def test_Graph():
    fig, ax = plt.subplots()
    g = Graph([(0, 1)], ax=ax)
    return fig
