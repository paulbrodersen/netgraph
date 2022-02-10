#!/usr/bin/env python
"""
Run tests.
"""

import numpy as np
import matplotlib.pyplot as plt
import pytest

from netgraph._artists import (
    PathPatchDataUnits,
    NodeArtist,
    EdgeArtist,
)
from netgraph._utils import _bspline


# set random seed for reproducibility
np.random.seed(42)

@pytest.mark.mpl_image_compare
def test_PathPatchDataUnits():

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)

    origin = (0, 0)
    width = 1
    height = 2
    lw = 0.25

    outer = plt.Rectangle((origin[0],    origin[1]),    width,      height,      facecolor='darkblue',  zorder=1)
    inner = plt.Rectangle((origin[0]+lw, origin[1]+lw), width-2*lw, height-2*lw, facecolor='lightblue', zorder=2)

    ax1.add_patch(outer)
    ax1.add_patch(inner)
    ax1.axis([-0.5, 1.5, -0.5, 2.5])
    ax1.set_aspect('equal')
    ax1.set_title('Desired')

    # create new patch with the adusted size, as the line is centered on the path
    mid = plt.Rectangle((origin[0]+lw/2, origin[1]+lw/2), width-lw, height-lw, facecolor='lightblue', zorder=1)
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

    outer = plt.Rectangle((origin[0],    origin[1]),    width,      height,      facecolor='darkblue',  zorder=1, linewidth=0)
    inner = plt.Rectangle((origin[0]+lw, origin[1]+lw), width-2*lw, height-2*lw, facecolor='lightblue', zorder=2, linewidth=0)

    ax1.add_patch(outer)
    ax1.add_patch(inner)
    ax1.axis([-0.1, 2.1, -0.1, 2.1])
    ax1.set_aspect('equal')
    ax1.set_title('Desired')

    # create new patch with the adusted size, as the line is centered on the path
    rp = NodeArtist(shape='s', xy=(1., 1.), radius=np.sqrt(2), facecolor='lightblue', edgecolor='darkblue', linewidth=lw)
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

    outer = plt.Circle(origin, radius,      facecolor='darkblue',  zorder=1, linewidth=0)
    inner = plt.Circle(origin, radius - lw, facecolor='lightblue', zorder=2, linewidth=0)

    ax1.add_patch(outer)
    ax1.add_patch(inner)
    ax1.axis([-0.1, 4.1, -0.1, 4.1])
    ax1.set_aspect('equal')
    ax1.set_title('Desired')

    # create new patch with the adusted size, as the line is centered on the path
    c = NodeArtist(shape='o', xy=origin, radius=radius, facecolor='lightblue', edgecolor='darkblue', linewidth=lw)
    ax2.add_patch(c)
    ax2.set_aspect('equal')
    ax2.set_title('Actual')

    return fig


@pytest.mark.mpl_image_compare
def test_simple_line():
    x = np.linspace(-1, 1, 1000)
    y = np.sqrt(1. - x**2)
    return plot_edge(x, y)


@pytest.mark.mpl_image_compare
def test_complicated_line():
    random_points = np.random.rand(5, 2)
    x, y = _bspline(random_points, n=1000).T
    return plot_edge(x, y)


@pytest.mark.mpl_image_compare
def test_left_arrow():
    x = np.linspace(-1, 1, 1000)
    y = np.sqrt(1. - x**2)
    return plot_edge(x, y, shape='left')


@pytest.mark.mpl_image_compare
def test_right_arrow():
    x = np.linspace(-1, 1, 1000)
    y = np.sqrt(1. - x**2)
    return plot_edge(x, y, shape='right')


def plot_edge(x, y, shape='full'):
    midline = np.c_[x, y]
    arrow = EdgeArtist(midline,
                       width       = 0.05,
                       head_width  = 0.1,
                       head_length = 0.15,
                       offset      = 0.1,
                       facecolor   = 'red',
                       edgecolor   = 'black',
                       linewidth   = 0.005,
                       alpha       = 0.5,
                       shape       = shape,
    )
    fig, ax = plt.subplots(1,1)
    ax.add_patch(arrow)
    ax.plot(x, y, color='black', alpha=0.1) # plot path for reference
    ax.set_aspect("equal")
    return fig


@pytest.mark.mpl_image_compare
def test_update_width():
    x = np.linspace(-1, 1, 1000)
    y = np.sqrt(1. - x**2)
    midline = np.c_[x, y]
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    for ax in axes:
        arrow = EdgeArtist(midline,
                           width       = 0.05,
                           head_width  = 0.1,
                           head_length = 0.15,
                           offset      = 0.1,
                           facecolor   = 'red',
                           edgecolor   = 'black',
                           linewidth   = 0.005,
                           alpha       = 0.5,
                           shape       = 'full',
        )
        ax.add_patch(arrow)
        ax.set_aspect("equal")
    arrow.update_width(0.1)
    fig.canvas.draw()
    axes[0].set_title('Before')
    axes[1].set_title('After')
    return fig
