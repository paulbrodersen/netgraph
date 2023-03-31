#!/usr/bin/env python
"""
Classes for artists used to display
- nodes / vertices
- edges
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.path import Path
from matplotlib.patches import PathPatch, transforms

from ._utils import (
    _get_parallel_line,
    _get_orthogonal_unit_vector,
    _shorten_line_by,
    _get_text_object_bbox,
    _get_text_object_dimensions,
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
    """NodeArtist base class.

    Parameters
    ----------
    shape : str or matplotlib.Path instance
        The shape of the node.
        If a string, the specification is as for matplotlib.scatter marker, i.e. one of 'so^>v<dph8'.
        If a path, the path has to be closed and the vertices have to be centered on (0,0).
    xy : tuple
        The (float x, float y) coordinates of the centroid.
    size : float
        The distance from the center to each of the vertices.
    **kwargs
        `Patch` properties:
        %(Patch_kwdoc)s

    See also
    --------
    CircularNodeArtist, RegularPolygonArtist

    """

    def __init__(self, path, xy, size, orientation=0, linewidth_correction=2, **kwargs):
        self._path = path
        self.xy = xy
        self.size = size
        self.orientation = orientation
        self.linewidth_correction = linewidth_correction
        self._patch_transform = transforms.Affine2D()
        super().__init__(path=self._path, **kwargs)
        self.transformed_path = self._path.transformed(self.get_patch_transform())

    def get_patch_transform(self):
        return self._patch_transform.clear() \
            .scale(self.size-self._lw_data/self.linewidth_correction) \
            .rotate(self.orientation) \
            .translate(*self.xy)

    def get_head_offset(self, edge_path):
        # Determine edge path points that are within node shape path.
        is_overlapping = self._path.contains_points(edge_path, transform=self.get_patch_transform())

        # Resample segment consisting of last point that is not enclosed and first point that is.
        idx = np.where(is_overlapping)[0][0]
        segment = edge_path[[idx-1, idx]]
        x, y = segment.T
        resampled = np.c_[np.linspace(x[0], x[1], 100), np.linspace(y[0], y[1], 100)]

        # Determine last resampled point that is not enclosed and compute distance to center.
        is_overlapping = self._path.contains_points(resampled, transform=self.get_patch_transform())
        idx = np.where(is_overlapping)[0][0] - 1
        offset = np.linalg.norm(edge_path[-1] - resampled[idx])
        return offset

    def get_tail_offset(self, edge_path):
        return self.get_head_offset(edge_path[::-1])

    def get_maximum_fontsize(self, label, ax, minimum=0, maximum=100, **font_dict):
        # NB: code assumes that
        # - fig.canvas.draw() has been called at least once
        # - fontdict contains parameters verticalalignment/va and horizontalalignment/ha and both are set to 'center'

        x, y = self.xy

        if 'size' in font_dict:
            size = font_dict['size']
            font_dict = dict(font_dict) # shallow copy
            del font_dict['size']
        elif 'fontsize' in font_dict:
            size = font_dict['fontsize']
            font_dict = dict(font_dict) # shallow copy
            del font_dict['fontsize']
        else:
            size = minimum + (maximum - minimum) / 2

        # binary search
        for _ in range(10):
            # bbox = _get_text_object_bbox(ax, x, y, label, ha='center', va='center', fontsize=size)
            bbox = _get_text_object_bbox(ax, x, y, label, fontsize=size, **font_dict)
            if self.transformed_path.intersects_bbox(bbox, filled=True) \
               and not self.transformed_path.intersects_bbox(bbox, filled=False): # i.e. label fully enclosed
                minimum = size
            else:
                maximum = size
            size = minimum + (maximum - minimum) / 2

        return size


class RegularPolygonNodeArtist(NodeArtist):
    """Instantiates a regular polygon node artist.

    Parameters
    ----------
    total_vertices : int
        Number of corners.
    orientation : float
        Orientation of the polygon in radians.
    xy : tuple
        The (float x, float y) coordinates of the centroid.
    size : float
        The distance from the center to each of the vertices.
    **kwargs
        `Patch` properties:
        %(Patch_kwdoc)s

    See also
    --------
    NodeArtist, CircularNodeArtist

    """

    def __init__(self, total_vertices, orientation, xy, size, **kwargs):
        path = Path.unit_regular_polygon(total_vertices)
        linewidth_correction = 2 * np.sin(np.pi/total_vertices) # derives from the ratio between a side and the radius in a regular polygon.
        super().__init__(path, xy, size, orientation=orientation, linewidth_correction=linewidth_correction, **kwargs)


class CircularNodeArtist(NodeArtist):
    """Instantiates a circular node artist.

    Parameters
    ----------
    xy : tuple
        The (float x, float y) coordinates of the centroid.
    size : float
        The distance from the center to each of the vertices.
    **kwargs
        `Patch` properties:
        %(Patch_kwdoc)s

    See also
    --------
    NodeArtist, RegularPolygonArtist

    """

    def __init__(self, xy, size, **kwargs):
        path = Path.circle()
        super().__init__(path, xy, size, **kwargs)

    def get_head_offset(self, edge_path):
        return self.size

    def get_tail_offset(self, edge_path):
        return self.size

    def get_maximum_fontsize(self, label, ax, **font_dict):
        diameter = 2 * (self.size - self._lw_data/self.linewidth_correction)
        width, height = _get_text_object_dimensions(ax, label, **font_dict)
        rescale_factor = diameter / np.sqrt(width**2 + height**2)

        if 'size' in font_dict:
            size = rescale_factor * font_dict['size']
        elif 'fontsize' in font_dict:
            size = rescale_factor * font_dict['fontsize']
        else:
            size = rescale_factor * plt.rcParams['font.size']

        return size


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
    head_offset : float, default 0.
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
                 head_offset = 0.,
                 tail_offset = 0.,
                 shape       = 'full',
                 curved      = False,
                 *args, **kwargs):

        # book keeping
        self.midline     = midline
        self.width       = width
        self.head_width  = head_width
        self.head_length = head_length
        self.head_offset = head_offset
        self.tail_offset = tail_offset
        self.shape       = shape
        self.curved      = curved

        self._update_path() # sets self._path
        super().__init__(self._path, *args, **kwargs)


    def _update_path(self):
        # Determine the actual start (and hence midline) of the arrow given the tail offsets;
        # assume an ordered midline from source to target, i.e. from arrow base to arrow head.
        arrow_midline      = _shorten_line_by(self.midline[::-1], self.tail_offset)[::-1]

        # Determine the actual endpoint (and hence midline) of the arrow given the offsets.
        arrow_midline      = _shorten_line_by(arrow_midline, self.head_offset)
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
