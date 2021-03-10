#!/usr/bin/env python
"""
Run tests.
"""

import numpy as np
import matplotlib.pyplot as plt
import pytest

from matplotlib.patches import Rectangle, Circle
from netgraph._artists import (
    PathPatchDataUnits,
    RegularPolygonDataUnits,
    CircleDataUnits,
)
@pytest.mark.mpl_image_compare
def test_PathPatchDataUnits():

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)

    origin = (0, 0)
    width = 1
    height = 2
    lw = 0.25

    outer = Rectangle((origin[0],    origin[1]),    width,      height,      facecolor='darkblue',  zorder=1)
    inner = Rectangle((origin[0]+lw, origin[1]+lw), width-2*lw, height-2*lw, facecolor='lightblue', zorder=2)

    ax1.add_patch(outer)
    ax1.add_patch(inner)
    ax1.axis([-0.5, 1.5, -0.5, 2.5])
    ax1.set_aspect('equal')
    ax1.set_title('Desired')

    # create new patch with the adusted size, as the line is centered on the path
    mid = Rectangle((origin[0]+lw/2, origin[1]+lw/2), width-lw, height-lw, facecolor='lightblue', zorder=1)
    path = mid.get_path().transformed(mid.get_patch_transform())
    pathpatch = PathPatchDataUnits(path, facecolor='lightblue', edgecolor='darkblue', linewidth=lw)
    ax2.add_patch(pathpatch)
    ax2.set_aspect('equal')
    ax2.set_title('Actual')

    return fig


@pytest.mark.mpl_image_compare
def test_RegularPolygonDataUnits():

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)

    origin = (0, 0)
    width = 2
    height = 2
    lw = 0.25

    outer = Rectangle((origin[0],    origin[1]),    width,      height,      facecolor='darkblue',  zorder=1, linewidth=0)
    inner = Rectangle((origin[0]+lw, origin[1]+lw), width-2*lw, height-2*lw, facecolor='lightblue', zorder=2, linewidth=0)

    ax1.add_patch(outer)
    ax1.add_patch(inner)
    ax1.axis([-0.1, 2.1, -0.1, 2.1])
    ax1.set_aspect('equal')
    ax1.set_title('Desired')

    # create new patch with the adusted size, as the line is centered on the path
    rp = RegularPolygonDataUnits((1., 1.), numVertices=4, radius=np.sqrt(2), orientation=np.pi/4, facecolor='lightblue', edgecolor='darkblue', linewidth=lw)
    ax2.add_patch(rp)
    ax2.set_aspect('equal')
    ax2.set_title('Actual')

    return fig


@pytest.mark.mpl_image_compare
def test_CircleDataUnits():

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)

    origin = (2, 2)
    radius = 2
    lw = 0.25

    outer = Circle(origin, radius,      facecolor='darkblue',  zorder=1, linewidth=0)
    inner = Circle(origin, radius - lw, facecolor='lightblue', zorder=2, linewidth=0)

    ax1.add_patch(outer)
    ax1.add_patch(inner)
    ax1.axis([-0.1, 4.1, -0.1, 4.1])
    ax1.set_aspect('equal')
    ax1.set_title('Desired')

    # create new patch with the adusted size, as the line is centered on the path
    c = CircleDataUnits(origin, radius=radius, facecolor='lightblue', edgecolor='darkblue', linewidth=lw)
    ax2.add_patch(c)
    ax2.set_aspect('equal')
    ax2.set_title('Actual')

    return fig


