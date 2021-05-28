#!/usr/bin/env python
import numpy as np

from scipy.interpolate import BSpline


def _save_cast_float_to_int(num):
    if isinstance(num, (float, int)) and np.isclose(num, int(num)):
        return int(num)
    return num


def _get_unique_nodes(edges):
    """
    Using numpy.unique promotes nodes to numpy.float/numpy.int/numpy.str,
    and breaks for nodes that have a more complicated type such as a tuple.
    """
    return list(set(_flatten(edges)))


def _flatten(nested_list):
    return [item for sublist in nested_list for item in sublist]


def _edge_list_to_adjacency_matrix(edges, edge_weights=None, unique_nodes=None):

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
    return [(source, target) for source, target in edges \
            if (source in nodes) and (target in nodes)]


def _bspline(cv, n=100, degree=5, periodic=False):
    """ Calculate n samples on a bspline

        cv :      Array of control vertices
        n  :      Number of samples to return
        degree:   Curve degree
        periodic: True - Curve is closed

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
    """Angle of vector in 2D."""
    angle = np.arctan2(dy, dx)
    if radians:
        angle *= 360 / (2.0 * np.pi)
    return angle


def _get_interior_angle_between(v1, v2, radians=False):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793

    Adapted from:
    https://stackoverflow.com/a/13849249/2912349
    """
    v1_u = get_unit_vector(v1)
    v2_u = get_unit_vector(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    if radians:
        angle *= 360 / (2 * np.pi)
    return angle


def get_unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def _get_signed_angle_between(v1, v2, radians=False):
    """
    Compute the signed angle in radians between two vectors.

    Adapted from:
    https://stackoverflow.com/a/16544330/2912349
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
    angles = np.linspace(0, 2*np.pi, n + 1)[:-1]
    angles = (angles + start_angle) % (2*np.pi)
    positions = np.array([_get_point_on_a_circle(xy, radius, angle) for angle in angles])
    return positions


def _get_point_on_a_circle(origin, radius, angle):
    x0, y0 = origin
    x = x0 + radius * np.cos(angle)
    y = y0 + radius * np.sin(angle)
    return np.array([x, y])


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


def _get_orthogonal_projection_onto_segment(point, segment):
    # Adapted from https://stackoverflow.com/a/61343727/2912349

    p1, p2 = segment

    segment_length = np.sum((p1-p2)**2)

    # The line extending the segment is parameterized as p1 + t (p2 - p1).
    # The projection falls where t = [(point-p1) . (p2-p1)] / |p2-p1|^2

    # Project onto line through p1 and p2.
    t = np.sum((point - p1) * (p2 - p1)) / segment_length

    # # Project onto line segment between p1 and p2 or closest point of the line segment.
    # t = max(0, t)

    return p1 + t * (p2 - p1)


def _get_text_object_dimensions(ax, string, *args, **kwargs):
    text_object = ax.text(0., 0., string, *args, **kwargs)
    renderer = _find_renderer(text_object.get_figure())
    bbox_in_display_coordinates = text_object.get_window_extent(renderer)
    bbox_in_data_coordinates = bbox_in_display_coordinates.transformed(ax.transData.inverted())
    w, h = bbox_in_data_coordinates.width, bbox_in_data_coordinates.height
    text_object.remove()
    return w, h


def _find_renderer(fig):
    """
    https://stackoverflow.com/questions/22667224/matplotlib-get-text-bounding-box-independent-of-backend
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
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.get_figure().set_facecolor('w')
    ax.set_frame_on(False)
    ax.get_figure().canvas.draw()


def _rank(vec):
    tmp = np.argsort(vec)
    ranks = np.empty_like(vec)
    ranks[tmp] = np.arange(len(vec))
    return ranks


def _invert_dict(mydict):
    inverse = dict()
    for key, value in mydict.items():
        inverse.setdefault(value, set()).add(key)
    return inverse
