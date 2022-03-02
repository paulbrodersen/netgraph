#!/usr/bin/env python
"""
Classes for artists used to display
- nodes / vertices
- edges
"""

import numpy as np

from matplotlib.path import Path
from matplotlib.patches import PathPatch, transforms

from ._utils import (
    _get_parallel_line,
    _get_orthogonal_unit_vector,
    _shorten_line_by,
)


class PathPatchDataUnits(PathPatch):
    """PathPatch in which the linewidth is also given in data units.

    Parameters
    ----------
    *args, **kwargs
        All arguments are passed through to PathPatch.

    Notes
    -----
    Adapted from https://stackoverflow.com/a/42972469/2912349

    """

    def __init__(self, *args, **kwargs):
        _lw_data = kwargs.pop("linewidth", 1)
        super().__init__(*args, **kwargs)
        self._lw_data = _lw_data

    def _get_lw(self):
        if self.axes is not None:
            ppd = 72./self.axes.figure.dpi
            trans = self.axes.transData.transform
            return ((trans((self._lw_data, self._lw_data))-trans((0, 0)))*ppd)[0]
            # return ((trans((self._lw_data, self._lw_data))-trans((0, 0)))*ppd)[1]
        else:
            return 1

    def _set_lw(self, lw):
        self._lw_data = lw

    _linewidth = property(_get_lw, _set_lw)


class NodeArtist(PathPatchDataUnits):
    """Implements the node artists class.

    Parameters
    ----------
    shape : str
        The shape of the node. Specification is as for matplotlib.scatter marker, i.e. one of 'so^>v<dph8'.
    xy : tuple
        The (float x, float y) coordinates of the centroid.
    radius : float
        The distance from the center to each of the vertices.
    **kwargs
        `Patch` properties:
        %(Patch_kwdoc)s

    """

    def __init__(self, shape, xy, radius, **kwargs):
        self.shape = shape
        self.xy = xy
        self.radius = radius

        if shape == 'o': # circle
            self.numVertices = None
            self.orientation = 0
        elif shape == '^': # triangle up
            self.numVertices = 3
            self.orientation = 0
        elif shape == '<': # triangle left
            self.numVertices = 3
            self.orientation = np.pi*0.5
        elif shape == 'v': # triangle down
            self.numVertices = 3
            self.orientation = np.pi
        elif shape == '>': # triangle right
            self.numVertices = 3
            self.orientation = np.pi*1.5
        elif shape == 's': # square
            self.numVertices = 4
            self.orientation = np.pi*0.25
        elif shape == 'd': # diamond
            self.numVertices = 4
            self.orientation = np.pi*0.5
        elif shape == 'p': # pentagon
            self.numVertices = 5
            self.orientation = 0
        elif shape == 'h': # hexagon
            self.numVertices = 6
            self.orientation = 0
        elif shape == '8': # octagon
            self.numVertices = 8
            self.orientation = 0
        else:
            raise ValueError("Node shape should be one of: 'so^>v<dph8'. Current shape:{}".format(shape))

        if self.shape == 'o': # circle
            self.linewidth_correction = 2
            self._path = Path.circle()
        else: # regular polygon
            self.linewidth_correction = 2 * np.sin(np.pi/self.numVertices) # derives from the ratio between a side and the radius in a regular polygon.
            self._path = Path.unit_regular_polygon(self.numVertices)

        self._patch_transform = transforms.Affine2D()
        super().__init__(path=self._path, **kwargs)

    def get_path(self):
        return self._path

    def get_patch_transform(self):
        # The factor 2 * sin(pi/n) d
        return self._patch_transform.clear() \
            .scale(self.radius-self._lw_data/self.linewidth_correction) \
            .rotate(self.orientation) \
            .translate(*self.xy)


class EdgeArtist(PathPatchDataUnits):
    """Implements the edge artist class.

    Parameters
    ----------
    midline : ndarray
        Array of (float x, float y) coordinates denoting the edge route.
    width : float, default 0.05
        The width of the edge (if shape is 'full').
    head_width : float, default 0.10
        Width of the arrow head.
        Set to a value close to zero (but not zero) to suppress drawing of arrowheads.
    head_length : float, default 0.15
        Length of the arrow head.
        Set to a value close to zero (but not zero) to suppress drawing of arrowheads.
    offset : float, default 0.
        For non-zero offset values, the end of the edge is offset from the target node.
        The distance is calculated along the midline.
    shape : {'full', 'left', 'right'}, default 'full'
        The shape of the arrow.
        For shapes 'left' and 'right' the arrow only one half of the arrow is plotted.
    curved : bool, default False
        Indicates if the midline is straight (False) or curved (True).

    """
    def __init__(self, midline,
                 width       = 0.05,
                 head_width  = 0.10,
                 head_length = 0.15,
                 offset      = 0.,
                 shape       = 'full',
                 curved      = False,
                 *args, **kwargs):

        # book keeping
        self.midline     = midline
        self.width       = width
        self.head_width  = head_width
        self.head_length = head_length
        self.shape       = shape
        self.offset      = offset
        self.curved      = curved

        self._update_path() # sets self._path
        super().__init__(self._path, *args, **kwargs)


    def _update_path(self):
        # Determine the actual endpoint (and hence midline) of the arrow given the offset;
        # assume an ordered midline from source to target, i.e. from arrow base to arrow head.
        arrow_midline      = _shorten_line_by(self.midline, self.offset)
        arrow_tail_midline = _shorten_line_by(arrow_midline, self.head_length)

        head_vertex_tip  = arrow_midline[-1]
        head_vertex_base = arrow_tail_midline[-1]
        (dx, dy), = _get_orthogonal_unit_vector(np.atleast_2d(head_vertex_tip - head_vertex_base)) * self.head_width / 2.

        if self.shape == 'full':
            tail_vertices_right = _get_parallel_line(arrow_tail_midline, -self.width / 2.)
            tail_vertices_left  = _get_parallel_line(arrow_tail_midline,  self.width / 2.)
            head_vertex_right = head_vertex_base - np.array([dx, dy])
            head_vertex_left  = head_vertex_base + np.array([dx, dy])

            vertices = np.concatenate([
                tail_vertices_right[::-1],
                tail_vertices_left,
                head_vertex_left[np.newaxis,:],
                head_vertex_tip[np.newaxis,:],
                head_vertex_right[np.newaxis,:],
                tail_vertices_right[np.newaxis,-1],
            ])
            codes = np.concatenate([
                [Path.MOVETO] + [Path.LINETO for _ in tail_vertices_right[1:]],
                [Path.LINETO for _ in tail_vertices_left],
                [Path.LINETO], # head_vertex_left
                [Path.LINETO], # head_vertex_tip
                [Path.LINETO], # head_vertex_right
                [Path.CLOSEPOLY] # tail_vertices_right[-1]
            ])

        elif self.shape == 'right':
            # tail_vertices_right = _get_parallel_line(arrow_tail_midline, -self.width / 2.)
            tail_vertices_right = _get_parallel_line(arrow_tail_midline, -0.6 * self.width)
            arrow_tail_midline = _get_parallel_line(arrow_tail_midline, -0.1 * self.width)
            head_vertex_right  = head_vertex_base - np.array([dx, dy])

            vertices = np.concatenate([
                tail_vertices_right[::-1],
                arrow_tail_midline,
                head_vertex_tip[np.newaxis,:],
                head_vertex_right[np.newaxis,:],
                tail_vertices_right[np.newaxis,-1]
            ])
            codes = np.concatenate([
                [Path.MOVETO] + [Path.LINETO for _ in tail_vertices_right[1:]],
                [Path.LINETO for _ in arrow_tail_midline],
                [Path.LINETO], # head_vertex_tip
                [Path.LINETO], # head_vertex_right
                [Path.CLOSEPOLY] # tail_vertices_right[-1]
            ])

        elif self.shape == 'left':
            # tail_vertices_left = _get_parallel_line(arrow_tail_midline,  self.width / 2.)
            tail_vertices_left = _get_parallel_line(arrow_tail_midline,  0.6 * self.width)
            arrow_tail_midline = _get_parallel_line(arrow_tail_midline,  0.1 * self.width)
            head_vertex_left = head_vertex_base + np.array([dx, dy])

            vertices = np.concatenate([
                arrow_tail_midline[::-1],
                tail_vertices_left,
                head_vertex_left[np.newaxis,:],
                head_vertex_tip[np.newaxis,:],
                arrow_tail_midline[np.newaxis,-1],
            ])
            codes = np.concatenate([
                [Path.MOVETO] + [Path.LINETO for _ in arrow_tail_midline[1:]],
                [Path.LINETO for _ in tail_vertices_left],
                [Path.LINETO], # head_vertex_left
                [Path.LINETO], # head_vertex_tip
                [Path.CLOSEPOLY] # arrow_tail_midline[-1]
            ])

        else:
            raise ValueError("Argument 'shape' needs to one of: 'left', 'right', 'full', not '{}'.".format(self.shape))

        self._path = Path(vertices, codes)


    def update_midline(self, midline):
        """Update the midline and recompute the edge path."""
        self.midline = midline
        self._update_path()


    def update_width(self, width, arrow=True):
        """
        Adjust the edge width. If arrow is True, the arrow head length and
        width are rescaled by the ratio new width / old width.
        """
        if arrow:
            ratio = width / self.width
            self.head_length *= ratio
            self.head_width *= ratio
        self.width = width
        self._update_path()
