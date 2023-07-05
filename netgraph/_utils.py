#!/usr/bin/env python
# coding: utf-8
"""
Netgraph utility functions.
"""

import numpy as np
import matplotlib as mpl

from numpy.linalg import matrix_rank
from scipy.interpolate import BSpline
from matplotlib.path import Path
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d


def _save_cast_float_to_int(num):
    """Cast number to an integer if it is close to an integer."""
    if isinstance(num, (float, int)) and np.isclose(num, int(num)):
        return int(num)
    return num


def _get_unique_nodes(edges):
    """
    Parameters
    ----------
    edges: list of tuple
        Edge list of the graph.

    Returns
    -------
    nodes: list
        List of unique nodes.

    Notes
    -----
    We cannot use numpy.unique, as it promotes nodes to numpy.float/numpy.int/numpy.str,
    and breaks for nodes that have a more complicated type such as a tuple.

    """
    return list(set(_flatten(edges)))


def _flatten(nested_list):
    """Flatten a nested list."""
    return [item for sublist in nested_list for item in sublist]


def _edge_list_to_adjacency_matrix(edges, edge_weights=None, unique_nodes=None):
    """Convert an edge list representation of a graph into the corresponding full rank adjacency matrix.

    Parameters
    ----------
    edges : list of tuple
        List of edges; each edge is identified by a (v1, v2) node tuple.
    edge_weights : list of int or float, optional
        List of corresponding edge weights.
    unique_nodes : list
        List of unique nodes. Required if any node is unconnected.

    Returns
    -------
    adjacency_matrix : numpy.array
        The full rank adjacency/weight matrix.

    """

    sources = [s for (s, _) in edges]
    targets = [t for (_, t) in edges]
    if edge_weights:
        weights = [edge_weights[edge] for edge in edges]
    else:
        weights = np.ones((len(edges)))

    if unique_nodes is None:
        # map nodes to consecutive integers
        nodes = sources + targets
        unique_nodes = set(nodes)

    indices = range(len(unique_nodes))
    node_to_idx = dict(zip(unique_nodes, indices))

    source_indices = [node_to_idx[source] for source in sources]
    target_indices = [node_to_idx[target] for target in targets]

    total_nodes = len(unique_nodes)
    adjacency_matrix = np.zeros((total_nodes, total_nodes))
    adjacency_matrix[source_indices, target_indices] = weights

    return adjacency_matrix


def _edge_list_to_adjacency_list(edges, directed=True):
    """Convert an edge list representation of a unweighted graph into the corresponding adjacency list representation.

    Parameters
    ----------
    edges : list of tuple
        List of edges; each edge is identified by a (node, node) tuple.
    directed : bool, default True
        Indicates if the graph is directed.

    Returns
    -------
    adjacency : dict node : set of nodes
        Dictionary mapping nodes to their set of connected neighbours.

    """
    if not directed:
        edges = edges + [(target, source) for (source, target) in edges] # forces copy

    adjacency = dict()
    for source, target in edges:
        if source in adjacency:
            adjacency[source] |= set([target])
        else:
            adjacency[source] = set([target])
    return adjacency


def _get_subgraph(edges, nodes):
    """Induce the subgraph using the specified nodes.

    Parameters
    ----------
    edges : list of tuple
        List of edges; each edge is identified by a (v1, v2) node tuple.
    nodes : list
        List of nodes.

    Returns
    -------
    subgraph_edges : list of tuple
        List of edges present in the subgraph.

    """

    return [(source, target) for source, target in edges \
            if (source in nodes) and (target in nodes)]


def _bspline(cv, n=100, degree=5, periodic=False):
    """Calculate n samples on a bspline.

    Parameters
    ----------
    cv : numpy.array
        Array of (x, y) control vertices.
    n : int
        Number of samples to return.
    degree : int
        Curve degree
    periodic : bool, default True
        If True, the curve is closed.

    Returns
    -------
    numpy.array
        Array of (x, y) spline vertices.

    Notes
    -----
    Adapted from https://stackoverflow.com/a/35007804/2912349

    """

    cv = np.asarray(cv)
    count = cv.shape[0]

    # Closed curve
    if periodic:
        kv = np.arange(-degree,count+degree+1)
        factor, fraction = divmod(count+degree+1, count)
        cv = np.roll(np.concatenate((cv,) * factor + (cv[:fraction],)),-1,axis=0)
        degree = np.clip(degree,1,degree)

    # Opened curve
    else:
        degree = np.clip(degree,1,count-1)
        kv = np.clip(np.arange(count+degree+1)-degree,0,count-degree)

    # Return samples
    max_param = count - (degree * (1-periodic))
    spl = BSpline(kv, cv, degree)
    return spl(np.linspace(0,max_param,n))


def _get_angle(dx, dy, radians=False):
    """Angle of a vector in 2D."""
    angle = np.arctan2(dy, dx)
    if radians:
        angle *= 360 / (2.0 * np.pi)
    return angle


def _get_interior_angle_between(v1, v2, radians=False):
    """Returns the interior angle between vectors v1 and v2.

    Parameters
    ----------
    v1, v2 : numpy.array
        The vectors in question.
    radians : bool, default False
        If True, return the angle in radians (otherwise it is in degrees).

    Returns
    -------
    angle : float
        The interior angle between two vectors.

    Examples
    --------
    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793

    Notes
    -----
    Adapted from https://stackoverflow.com/a/13849249/2912349

    See also
    --------
    _get_signed_angle_between

    """

    v1_u = _get_unit_vector(v1)
    v2_u = _get_unit_vector(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    if radians:
        angle *= 360 / (2 * np.pi)
    return angle


def _get_unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def _get_signed_angle_between(v1, v2, radians=False):
    """Returns the signed angle between vectors v1 and v2.

    Parameters
    ----------
    v1, v2 : numpy.array
        The vectors in question.
    radians : bool, default False
        If True, return the angle in radians (otherwise it is in degrees).

    Returns
    -------
    angle : float
        The signed angle between two vectors.

    Notes
    -----
    Adapted from https://stackoverflow.com/a/16544330/2912349

    See also
    --------
    _get_interior_angle_between

    """

    x1, y1 = v1
    x2, y2 = v2
    dot = x1*x2 + y1*y2
    det = x1*y2 - y1*x2
    angle = np.arctan2(det, dot)
    if radians:
        angle *= 360 / (2 * np.pi)
    return angle


def _get_n_points_on_a_circle(xy, radius, n, start_angle=0):
    """Determine the positions of n evenly spaced points on a circle with a given (x, y) origin and radius.

    Parameters
    ----------
    xy : tuple of float
        The origin of the circle.
    radius : float
        The radius of the circle.
    n : int
        The number of points returned.
    start_angle : float, default 0
        The angle at which the first point is placed in radians.

    Returns
    -------
    positions ; numpy.array
        Array of n (x, y) coordinates.

    """

    angles = np.linspace(0, 2*np.pi, n + 1)[:-1]
    angles = (angles + start_angle) % (2*np.pi)
    positions = np.array([_get_point_on_a_circle(xy, radius, angle) for angle in angles])
    return positions


def _get_point_on_a_circle(origin, radius, angle):
    """Compute the (x, y) coordinate of a point at a specified angle on a circle given by its (x, y) origin and radius."""

    x0, y0 = origin
    x = x0 + radius * np.cos(angle)
    y = y0 + radius * np.sin(angle)
    return np.array([x, y])


def _get_parallel_line(path, delta):
    """Compute a parallel to a given path with an offset delta.

    Parameters
    ----------
    path : numpy.array
        Array of (x, y) path coordinates.
    delta : float
        Offset from the path.

    Returns
    -------
    path : numpy.array
        Array of (x, y) coordinates corresponding to the parallel path.

    """

    # initialise output
    orthogonal_unit_vector = np.zeros_like(path)

    tangents = path[2:] - path[:-2] # using the central difference approximation
    orthogonal_unit_vector[1:-1] = _get_orthogonal_unit_vector(tangents)

    # handle start and end points
    orthogonal_unit_vector[ 0] = _get_orthogonal_unit_vector(np.atleast_2d([path[ 1] - path[ 0]]))
    orthogonal_unit_vector[-1] = _get_orthogonal_unit_vector(np.atleast_2d([path[-1] - path[-2]]))

    return path + delta * orthogonal_unit_vector


def _get_orthogonal_unit_vector(v):
    """Determine the orthogonal unit vector to a given vector.

    Parameters
    ----------
    v : numpy.array
        The input vector.

    Returns
    -------
    w : numpy.array
        The output vector.

    Notes
    -----
    Adapted from https://stackoverflow.com/a/16890776/2912349

    """
    if not np.all(np.isclose(v, 0)):
        v = v / np.linalg.norm(v, axis=-1)[:, None] # unit vector
        w = np.c_[-v[:,1], v[:,0]]                  # orthogonal vector
        w = w / np.linalg.norm(w, axis=-1)[:, None] # orthogonal unit vector
        return w
    else:
        return v


def _shorten_spline_by(path, distance):
    """Cut path off at the end by `distance`.

    Parameters
    ----------
    path : numpy.array
        Array of (x, y) path coordinates.
    distance : float
        Distance travelled along the path to remove from the end of the path.

    Returns
    -------
    numpy.array
        Array of (x, y) coordinates of the shortened path.

    """

    if distance > 0:
        distance_to_end = np.linalg.norm(path - path[-1], axis=1)
        is_valid = (distance_to_end - distance) >= 0
        if np.any(is_valid):
            idx = np.where(is_valid)[0][-1] # i.e. the last valid point
        else:
            idx = 0

        # We could truncate the  path using `path[:idx+1]` and return here.
        # However, if the path is not densely sampled, the error will be large.
        # Therefor, we compute a point that is on the line from the last valid point to
        # the end point, and append it to the truncated path.
        vector = path[idx] - path[-1]
        unit_vector = vector / np.linalg.norm(vector)
        new_end_point = path[-1] + distance * unit_vector

        return np.concatenate([path[:idx+1], new_end_point[None, :]], axis=0)

    else:
        return path


def _get_point_along_spline(spline, fraction):
    """Determine the point coordinates at a given fraction of the spline.

    Parameters
    ----------
    spline : numpy.array
        (N points, 2) array of (x, y) spline coordinates.
    fraction : float
        Fraction of the spline. Has to be a value between 0 and 1.

    Returns
    -------
    point : 2-tuple of floats
        The (x, y) point coordinates.

    """

    assert 0 <= fraction <= 1, "Fraction has to be a value between 0 and 1."
    deltas = np.diff(spline, axis=0)
    successive_distances = np.sqrt(np.sum(deltas**2, axis=1))
    cumulative_sum = np.cumsum(successive_distances)
    desired_length = cumulative_sum[-1] * fraction
    idx = np.where(cumulative_sum >= desired_length)[0][0] # upper bound
    overhang = cumulative_sum[idx] - desired_length
    x, y = spline[idx+1] - overhang/successive_distances[idx] * deltas[idx]
    return x, y


def _resample_spline(spline, total_samples=100):
    """Resample a spline using the given number of points.

    Parameters
    ----------
    spline : numpy.array
        (N points, 2) array of (x, y) spline coordinates.
    total_samples : int
        The number of evenly spaces points after re-sampling.

    Returns
    -------
    resampled : numpy.array
        (M samples, 2) array of (x, y) resampled spline coordinates.

    Notes
    -----
    Adapted from: https://stackoverflow.com/a/52020098/2912349
    """
    distance = np.cumsum(np.sqrt(np.sum(np.diff(spline, axis=0)**2, axis=1)))
    distance = np.insert(distance, 0, 0)/distance[-1]
    interpolator = interp1d(distance, spline, kind='slinear', axis=0)
    return interpolator(np.linspace(0, 1, total_samples))


def _get_tangent_at_point(spline, fraction):
    """Compute the tangent to a spline at a given fraction of the spline.

    Parameters
    ----------
    spline : numpy.array
        (N points, 2) array of (x, y) spline coordinates.
    fraction : float
        Fraction of the spline. Has to be a value between 0 and 1.

    Returns
    -------
    tangent : numpy.array
        The (dx, dy) tangent.

    """

    assert 0 <= fraction <= 1, "Fraction has to be a value between 0 and 1."
    deltas = np.diff(spline, axis=0)
    successive_distances = np.sqrt(np.sum(deltas**2, axis=1))
    cumulative_sum = np.cumsum(successive_distances)
    desired_length = cumulative_sum[-1] * fraction
    idx = np.where(cumulative_sum >= desired_length)[0][0] # upper bound
    return deltas[idx]


def _get_orthogonal_projection_onto_segment(point, segment):
    """Given a segment defined by points P1 and P2, determine the orthogonal projection of a point P3 onto said segment.

    Parameters
    ----------
    segment : tuple of (float x, float y) tuples
        The segment defined by (P1, P2).
    point : (float x, float y) tuple
        The point P3 to be projected onto the segment.

    Returns
    -------
    projected_point : (float x, float y)
        The coordinates of the projected point.

    Notes
    -----
    Adapted from https://stackoverflow.com/a/61343727/2912349

    """

    p1, p2 = segment

    segment_length = np.sum((p1-p2)**2)

    # The line extending the segment is parameterized as p1 + t (p2 - p1).
    # The projection falls where t = [(point-p1) . (p2-p1)] / |p2-p1|^2
    t = np.sum((point - p1) * (p2 - p1)) / segment_length

    # # Project onto line segment between p1 and p2 or closest point of the line segment.
    # t = max(0, t)

    return p1 + t * (p2 - p1)


def _find_renderer(fig):
    """
    Return the renderer for a given matplotlib figure.

    Notes
    -----
    Adapted from https://stackoverflow.com/a/22689498/2912349

    """

    if hasattr(fig.canvas, "get_renderer"):
        # Some backends, such as TkAgg, have the get_renderer method, which
        # makes this easy.
        renderer = fig.canvas.get_renderer()
    else:
        # Other backends do not have the get_renderer method, so we have a work
        # around to find the renderer. Print the figure to a temporary file
        # object, and then grab the renderer that was used.
        # (I stole this trick from the matplotlib backend_bases.py
        # print_figure() method.)
        import io
        fig.canvas.print_pdf(io.BytesIO())
        renderer = fig._cachedRenderer
    return(renderer)


def _make_pretty(ax):
    """Remove the figure frame, x- and y-ticks, and set the aspect to equal."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.get_figure().set_facecolor('w')
    # ax.set_frame_on(False) # also removes background
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.get_figure().canvas.draw()


def _rank(vec):
    """Compute the rank for each value in a vector."""
    tmp = np.argsort(vec)
    ranks = np.empty_like(vec)
    ranks[tmp] = np.arange(len(vec))
    return ranks


def _invert_dict(mydict):
    """Invert a dictionary such that values map to keys."""
    inverse = dict()
    for key, value in mydict.items():
        inverse.setdefault(value, set()).add(key)
    return inverse


def _get_connected_components(adjacency_list):
    """
    Get the connected components given a graph in adjacency list format.

    Parameters
    ---------
    adjacency_list : dict node ID : set of node IDs
        Adjacency list, i.e. a mapping from each node to its neighbours.

    Returns
    -------
    components : list of sets of node IDs
        The unconnected components of the graph.

    """

    components = []
    not_visited = set(list(adjacency_list.keys()))
    while not_visited: # i.e. while stack is non-empty (empty set is interpreted as `False`)
        start = not_visited.pop()
        component = _bfs(adjacency_list, start)
        components.append(component)

        #  remove nodes that are in the component that we just found
        for node in component:
            try:
                not_visited.remove(node)
            except KeyError:
                # KeyErrors occur when we try to remove
                # 1) the start node (which we already popped), or
                # 2) leaf nodes, i.e. nodes with no outgoing edges
                pass

    # Often, we are only interested in the largest component,
    # hence we return the list of components sorted by size, largest first.
    components = sorted(components, key=len, reverse=True)

    return components


def _dfs(adjacency_list, start, visited=None):
    """Depth first search on a given adjacency list.

    Parameters
    ----------
    adjacency_list : dict node ID : set of node IDs
        Adjacency list, i.e. a mapping from each node to its neighbours.

    start : node ID
        The starting node.

    visited : set of node IDs or None, default None
        Previously visited nodes.

    Returns
    -------
    visited : set of node IDs or None, default None
        Previously and newly visited nodes.

    """

    if visited is None:
        visited = set()
    visited.add(start)
    for node in adjacency_list[start] - visited:
        if node in adjacency_list:
            _dfs(adjacency_list, node, visited)
        else: # otherwise no outgoing edge
            visited.add(node)
    return visited


def _bfs(adjacency_list, start):
    """A fast BFS node generator.

    Parameters
    ----------
    adjacency_list : dict node ID : set of node IDs
        Adjacency list, i.e. a mapping from each node to its neighbours.
    start : node ID
        The starting node.

    Returns
    -------
    visited : set of node IDs or None, default None
        All reachable nodes from source.

    Notes
    -----
    Adapted from networkx.algorithms.components._plain_bfs
    """

    visited = {start}
    next_level = [start]
    while next_level:
        this_level = next_level
        next_level = []
        for node in this_level:
            for neighbour in adjacency_list[node]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    next_level.append(neighbour)
            if len(visited) == len(adjacency_list):
                return visited
    return visited


def _get_gradient_and_intercept(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    gradient = (y2 - y1) / (x2 - x1)
    intercept = (x2 * y1 - x1 * y2) / (x2 - x1)
    return gradient, intercept


def _is_above_line(points, gradient, intercept):
    """Returns true for points above a given line.

    Notes
    -----
    Adapted from https://stackoverflow.com/a/45769740/2912349

    """
    a = np.array([0, intercept])
    b = np.array([1, intercept + gradient])
    return np.cross(points-a, b-a) < 0


def _reflect_across_line(points, gradient, intercept):
    """Reflect a set of points across a given line.

    Notes
    -----
    Adapted from https://stackoverflow.com/a/45769740/2912349

    """
    x0, y0 = points.T
    d = (x0 + (y0 - intercept) * gradient) / (1 + gradient**2)
    x1 = 2 * d - x0
    y1 = 2 * d * gradient - y0 + 2 * intercept
    return np.c_[x1, y1]


def _are_collinear(points, tol=None):
    "Test if the given points are collinear."
    points = np.array(points)
    points -= points.mean(axis=0)[np.newaxis, :]
    rank = matrix_rank(points, tol=tol)
    return rank == 1


def _convert_polar_to_cartesian_coordinates(rho, phi):
    # Adapted from https://stackoverflow.com/a/26757297/2912349
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def _normalize_numeric_argument(numeric_or_dict, dict_keys, variable_name, allow_none=False):
    if allow_none:
        allowed_types = (int, float, type(None))
    else:
        allowed_types = (int, float)
    if isinstance(numeric_or_dict, allowed_types):
        return {key : numeric_or_dict for key in dict_keys}
    elif isinstance(numeric_or_dict, dict):
        _check_completeness(numeric_or_dict, dict_keys, variable_name)
        _check_types(numeric_or_dict.values(), allowed_types, variable_name)
        return numeric_or_dict
    else:
        msg = f"The type of {variable_name} has to be either a int, float, or a dict."
        msg += f"\nThe current type is {type(numeric_or_dict)}."
        raise TypeError(msg)


def _check_completeness(given_set, desired_set, variable_name):
    # ensure that iterables are sets
    # TODO: check that iterables can safely be converted to sets (unlike dict keys)
    given_set = set(given_set)
    desired_set = set(desired_set)

    complete = given_set.issuperset(desired_set)
    if not complete:
        missing = desired_set - given_set
        msg = f"{variable_name} is incomplete. The following elements are missing:"
        for item in missing:
            if isinstance(item, str):
                msg += f"\n\'{item}\'"
            else:
                msg += f"\n{item}"
        raise ValueError(msg)


def _check_types(items, types, variable_name):
    for item in items:
        if not isinstance(item, types):
            msg = f"Item {item} in {variable_name} is of the wrong type."
            msg += f"\nExpected type: {types}"
            msg += f"\nActual type: {type(item)}"
            raise TypeError(msg)


def _normalize_string_argument(str_or_dict, dict_keys, variable_name):
    if isinstance(str_or_dict, str):
        return {key : str_or_dict for key in dict_keys}
    elif isinstance(str_or_dict, dict):
        _check_completeness(set(str_or_dict), dict_keys, variable_name)
        _check_types(str_or_dict.values(), str, variable_name)
        return str_or_dict
    else:
        msg = f"The type of {variable_name} has to be either a str or a dict."
        msg += f"The current type is {type(str_or_dict)}."
        raise TypeError(msg)


def _normalize_shape_argument(str_path_or_dict, dict_keys, variable_name):
    if isinstance(str_path_or_dict, str):
        return {key : str_path_or_dict for key in dict_keys}
    elif isinstance(str_path_or_dict, Path):
        return {key : str_path_or_dict for key in dict_keys}
    elif isinstance(str_path_or_dict, dict):
        _check_completeness(set(str_path_or_dict), dict_keys, variable_name)
        _check_types(str_path_or_dict.values(), (str, Path), variable_name)
        return str_path_or_dict
    else:
        msg = f"The type of {variable_name} has to be either a str, matplotlib.path.Path or a dict."
        msg += f"The current type is {type(str_path_or_dict)}."
        raise TypeError(msg)


def _normalize_color_argument(color_or_dict, dict_keys, variable_name):
    if mpl.colors.is_color_like(color_or_dict):
        return {key : color_or_dict for key in dict_keys}
    elif color_or_dict is None:
        return {key : color_or_dict for key in dict_keys}
    elif isinstance(color_or_dict, dict):
        _check_completeness(set(color_or_dict), dict_keys, variable_name)
        # TODO: assert that each element is a valid color
        return color_or_dict
    else:
        msg = f"The type of {variable_name} has to be either a valid matplotlib color specification or a dict."
        raise TypeError(msg)


def _rescale_dict_values(mydict, scalar):
    return {key: value * scalar for (key, value) in mydict.items()}


# # Variant no 1: use force directed layout to determine a suitable node label placements
# # pros : labels repel each other
# # cons : does not work very well; the optimum placement can still result in a collision
# def _get_optimal_offsets(self, anchors, offsets, avoid, total_iterations=5):
#     # Compute the net repulsion exerted on each label by nodes, edges and other labels.
#     # Place the label in the direction of net repulsion at the desired distance from the corresponding node (anchor).
#     # TODO Test if gradually stepping in the direction of net repulsion improves results.
#     for ii in range(total_iterations):
#         repulsion = self._get_repulsion(anchors + offsets, avoid)
#         directions = repulsion / np.linalg.norm(repulsion, axis=-1)[:, np.newaxis]
#         offsets = np.linalg.norm(offsets, axis=-1)[:, np.newaxis] * directions
#     return offsets


# def _get_repulsion(self, mobile, fixed, minimum_distance=0.01):
#     combined = np.concatenate([mobile, fixed], axis=0)

#     delta = mobile[np.newaxis, :, :] - combined[:, np.newaxis, :]
#     distance = np.linalg.norm(delta, axis=-1)
#     direction = delta / distance[..., None] # i.e. the unit vector

#     # 1. We clip the distance as we want to reduce overlaps with
#     # all nearby plot elements, not just the one that overlaps the
#     # most.
#     # 2. We only care about interactions with nearby objects, so
#     # we heavily penalise repulsion from far away items by using a
#     # exponent.
#     magnitude = 1. / np.clip(distance, minimum_distance, np.inf)**6
#     repulsion = direction * magnitude[..., None]

#     for ii in range(repulsion.shape[-1]):
#         np.fill_diagonal(repulsion[:, :, ii], 0)

#     return np.sum(repulsion, axis=0)


# Variant no 2:
# pros : straightforward optimisation; works very well
# cons : labels can still collide with each other
def _get_optimal_offsets(anchors, offsets, avoid, total_queries_per_point=360):
    """Find the location at a specified distance (`offset`) away from a
    given other location (`anchor`) that is furthest away from all
    locations in `avoid`.
    """
    tree = cKDTree(avoid)
    output = np.zeros((len(offsets), 2))
    for ii, (anchor, offset) in enumerate(zip(anchors, offsets)):
        x = _get_n_points_on_a_circle(anchor, np.linalg.norm(offset), total_queries_per_point)
        # distances, _ = tree.query(x, 1) # can result in many ties; first element is arbitrarily chosen
        # output[ii] = x[np.argmax(distances)]
        distances, _ = tree.query(x, 2)
        output[ii] = x[np.argmax(np.sum(distances, axis=1))]
    return output - anchors


def _print_progress_bar(iteration, total_iterations, prefix='', suffix='', length=100):
    """Call in a loop to create terminal progress bar.

    Parameters
    ----------
    iteration: int
        The current iteration.
    total_iterations: int
        The total number of iterations.
    prefix: str (default '')
        The prefix.
    suffix: str (default '')
        The suffix.
    length: int (default 100)
        The character length of bar.

    Notes
    -----
    Adapted from: https://stackoverflow.com/a/34325723/2912349

    """
    fraction_complete = iteration / total_iterations
    bar = '█' * int(length * fraction_complete) + '-' * int(length * (1 - fraction_complete))
    print(f'\r{prefix} |{bar}| {100 * fraction_complete:.1f}% {suffix}', end="\r")
    if iteration == total_iterations:
        print()


def _get_total_pixels(fig):
    w, h = fig.get_size_inches()
    dpi = fig.get_dpi()
    total_pixels = w * h * dpi**2
    return total_pixels
