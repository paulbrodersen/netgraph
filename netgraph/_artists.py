#!/usr/bin/env python
"""
Classes for artists used to display
- nodes / vertices
- edges
"""

import numpy as np

from matplotlib.path import Path
from matplotlib.patches import PathPatch, transforms


class PathPatchDataUnits(PathPatch):
    # adapted from https://stackoverflow.com/a/42972469/2912349
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

    def __init__(self, shape, xy, radius, **kwargs):
        """
        Parameters
        ----------
        shape : string
            The shape of the node. Specification is as for matplotlib.scatter
            marker, i.e. one of 'so^>v<dph8'.
        xy : (float, float)
            The center position.
        radius : float
            The distance from the center to each of the vertices.
        **kwargs
            `Patch` properties:
            %(Patch_kwdoc)s
        """
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

        if self.shape is 'full':
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

        elif self.shape is 'right':
            tail_vertices_right = _get_parallel_line(arrow_tail_midline, -self.width / 2.)
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

        elif self.shape is 'left':
            tail_vertices_left = _get_parallel_line(arrow_tail_midline,  self.width / 2.)
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



def _get_parallel_line(path, delta):
    # initialise output
    orthogonal_unit_vector = np.zeros_like(path)

    tangents = path[2:] - path[:-2] # using the central difference approximation
    orthogonal_unit_vector[1:-1] = _get_orthogonal_unit_vector(tangents)

    # handle start and end points
    orthogonal_unit_vector[ 0] = _get_orthogonal_unit_vector(np.atleast_2d([path[ 1] - path[ 0]]))
    orthogonal_unit_vector[-1] = _get_orthogonal_unit_vector(np.atleast_2d([path[-1] - path[-2]]))

    return path + delta * orthogonal_unit_vector


def _get_orthogonal_unit_vector(v):
    # adapted from https://stackoverflow.com/a/16890776/2912349
    v = v / np.linalg.norm(v, axis=-1)[:, None] # unit vector
    w = np.c_[-v[:,1], v[:,0]]                  # orthogonal vector
    w = w / np.linalg.norm(w, axis=-1)[:, None] # orthogonal unit vector
    return w


def _shorten_line_by(path, distance):
    """
    Cut path off at the end by `distance`.
    """
    distance_to_end = np.linalg.norm(path - path[-1], axis=1)
    idx = np.where(distance_to_end - distance >= 0)[0][-1] # i.e. the last valid point

    # We could truncate the  path using `path[:idx+1]` and return here.
    # However, if the path is not densely sampled, the error will be large.
    # Therefor, we compute a point that is on the line from the last valid point to
    # the end point, and append it to the truncated path.
    vector = path[idx] - path[-1]
    unit_vector = vector / np.linalg.norm(vector)
    new_end_point = path[-1] + distance * unit_vector

    return np.concatenate([path[:idx+1], new_end_point[None, :]], axis=0)


def _get_point_along_spline(spline, fraction):
    assert 0 <= fraction <= 1, "Fraction has to be a value between 0 and 1."
    deltas = np.diff(spline, axis=0)
    successive_distances = np.sqrt(np.sum(deltas**2, axis=1))
    cumulative_sum = np.cumsum(successive_distances)
    desired_length = cumulative_sum[-1] * fraction
    idx = np.where(cumulative_sum >= desired_length)[0][0] # upper bound
    overhang = cumulative_sum[idx] - desired_length
    x, y = spline[idx+1] - overhang/successive_distances[idx] * deltas[idx]
    return x, y


def _get_tangent_at_point(spline, fraction):
    assert 0 <= fraction <= 1, "Fraction has to be a value between 0 and 1."
    deltas = np.diff(spline, axis=0)
    successive_distances = np.sqrt(np.sum(deltas**2, axis=1))
    cumulative_sum = np.cumsum(successive_distances)
    desired_length = cumulative_sum[-1] * fraction
    idx = np.where(cumulative_sum >= desired_length)[0][0] # upper bound
    return deltas[idx]
