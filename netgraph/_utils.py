#!/usr/bin/env python
# coding: utf-8
"""
Netgraph utility functions.
"""

import numpy as np

from numpy.linalg import matrix_rank
from scipy.interpolate import BSpline


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

    v = v / np.linalg.norm(v, axis=-1)[:, None] # unit vector
    w = np.c_[-v[:,1], v[:,0]]                  # orthogonal vector
    w = w / np.linalg.norm(w, axis=-1)[:, None] # orthogonal unit vector
    return w


def _shorten_line_by(path, distance):
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


def _get_text_object_dimensions(ax, string, *args, **kwargs):
    """Precompute the dimensions of a text object on a given axis in data coordinates.

    Parameters
    ----------
    ax : matplotlib.axis object
        The matplotlib axis.
    string : str
        The string.
    *args, **kwargs
        Passed to ax.text().

    Returns
    -------
    width, height : float
        The dimensions of the text box in data units.

    """

    text_object = ax.text(0., 0., string, *args, **kwargs)
    renderer = _find_renderer(text_object.get_figure())
    bbox_in_display_coordinates = text_object.get_window_extent(renderer)
    bbox_in_data_coordinates = bbox_in_display_coordinates.transformed(ax.transData.inverted())
    w, h = bbox_in_data_coordinates.width, bbox_in_data_coordinates.height
    text_object.remove()
    return w, h


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
    ax.set_frame_on(False)
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
        component = _dfs(adjacency_list, start)
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
