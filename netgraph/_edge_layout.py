import itertools
import warnings
import numpy as np

from uuid import uuid4

from ._utils import _bspline, _get_n_points_on_a_circle, _get_signed_angle_between
from ._node_layout import get_fruchterman_reingold_layout, _clip_to_frame

# for profiling with kernprof/line_profiler
try:
    profile
except NameError:
    profile = lambda x: x


def get_straight_edge_paths(edge_list, node_positions, edge_width):
    """Determine the edge layout, where edges are represented by straight
    lines connecting the source and target node. Bi-directional edges
    are offset from one another by one edge width.

    Arguments:
    ----------
    edge_list : list of (source node ID, target node ID) 2-tuples
        The edges.

    node_positions : dict node ID : (x, y) positions
        The node positions.

    edge_width: dict edge : float
        The width of each edge.

    Returns:
    --------
    edge_paths : dict edge : ndarray
        Dictionary mapping each edge to a list of edge segments.

    """
    edge_paths = dict()
    for (source, target) in edge_list:

        if source == target:
            msg = "Plotting of self-loops not supported for straight edges."
            msg += "Ignoring edge ({}, {}).".format(source, target)
            warnings.warn(msg)
            continue

        x1, y1 = node_positions[source]
        x2, y2 = node_positions[target]

        if (target, source) in edge_list: # i.e. bidirectional
            # shift edge to the right (looking along the arrow)
            x1, y1, x2, y2 = _shift_edge(x1, y1, x2, y2, delta=-0.5*edge_width[(source, target)])

        edge_paths[(source, target)] = np.c_[[x1, x2], [y1, y2]]

    return edge_paths


def _shift_edge(x1, y1, x2, y2, delta):
    # get orthogonal unit vector
    v = np.r_[x2-x1, y2-y1] # original
    v = np.r_[-v[1], v[0]] # orthogonal
    v = v / np.linalg.norm(v) # unit
    dx, dy = delta * v
    return x1+dx, y1+dy, x2+dx, y2+dy


def get_curved_edge_paths(edge_list, node_positions,
                          total_control_points_per_edge = 11,
                          selfloop_radius               = 0.1,
                          origin                        = np.array([0, 0]),
                          scale                         = np.array([1, 1]),
                          k                             = None,
                          initial_temperature           = 0.1,
                          total_iterations              = 50,
                          node_size                     = None
):

    """Determine the edge layout, where edges are represented by curved
    lines connecting the source and target node. Edges paths avoid
    nodes and each other. The edge layout is determined using the
    Fruchterman-Reingold algorithm.

    Arguments:
    ----------
    edge_list : list of (source node ID, target node ID) 2-tuples
        The edges.

    node_positions : dict node ID : (x, y) positions
        The node positions.

    selfloop_radius : float
        Self-loops are drawn as circles adjacent to a node. This value determine
        the radius of the circle.

    total_control_points_per_edge : int (default 11)
        Number of control

    k : float or None (default None)
        Expected mean segment length. If None, initialized to :
        sqrt(area / total nodes) / total control points + 1.

    origin : (float x, float y) tuple or None (default (0, 0))
        The lower left hand corner of the bounding box specifying the extent of the layout.

    scale : (float delta x, float delta y) or None (default (1, 1))
        The width and height of the bounding box specifying the extent of the layout.

    total_iterations : int (default 50)
        Number of iterations in the Fruchterman-Reingold algorithm.

    initial_temperature: float (default 1.)
        Temperature controls the maximum node displacement on each iteration.
        Temperature is decreased on each iteration to eventually force the algorithm
        into a particular solution. The size of the initial temperature determines how
        quickly that happens. Values should be much smaller than the values of `scale`.

    node_size : dict node ID : float
        Size of nodes. Used for node avoidance.

    Returns:
    --------
    edge_paths : dict edge : ndarray
        Dictionary mapping each edge to a list of edge segments.

    """

    expanded_edge_list, edge_to_control_points = _insert_control_points(
        edge_list, total_control_points_per_edge)

    control_point_positions = _initialize_control_point_positions(
        edge_to_control_points, node_positions, selfloop_radius, origin, scale)

    expanded_node_positions = _optimize_control_point_positions(
        expanded_edge_list, node_positions, control_point_positions, total_control_points_per_edge,
        origin, scale, k, initial_temperature, total_iterations, node_size,
    )

    edge_to_path = _fit_splines_through_control_points(
        edge_to_control_points, expanded_node_positions)

    return edge_to_path


def _insert_control_points(edge_list, total_control_points_per_edge=11):
    """
    Create a new, expanded edge list, in which each edge is split into multiple segments.
    There are total_control_points + 1 segments / edges for each original edge.
    """
    expanded_edge_list = []
    edge_to_control_points = dict()

    for source, target in edge_list:
        control_points = [uuid4() for _ in range(total_control_points_per_edge)]
        edge_to_control_points[(source, target)] = control_points

        sources = [source] + control_points
        targets = control_points + [target]
        expanded_edge_list.extend(zip(sources, targets))

    return expanded_edge_list, edge_to_control_points


def _initialize_control_point_positions(edge_to_control_points, node_positions,
                                        selfloop_radius = 0.1,
                                        origin          = np.array([0, 0]),
                                        scale           = np.array([1, 1])
):
    """
    Initialise the positions of the control points to positions on a straight line between source and target node.
    For self-loops, initialise the positions on a circle next to the node.
    """

    nonloops_to_control_points = {(source, target) : pts for (source, target), pts in edge_to_control_points.items() if source != target}
    selfloops_to_control_points = {(source, target) : pts for (source, target), pts in edge_to_control_points.items() if source == target}

    control_point_positions = dict()
    control_point_positions.update(_initialize_nonloops(nonloops_to_control_points, node_positions))
    control_point_positions.update(_initialize_selfloops(selfloops_to_control_points, node_positions, selfloop_radius, origin, scale))

    return control_point_positions


def _initialize_nonloops(edge_to_control_points, node_positions):
    control_point_positions = dict()
    for (source, target), control_points in edge_to_control_points.items():
        control_point_positions.update(_init_nonloop(source, target, control_points, node_positions))
    return control_point_positions


def _init_nonloop(source, target, control_points, node_positions):
    delta = node_positions[target] - node_positions[source]
    output = dict()
    for ii, control_point in enumerate(control_points):
        # y = mx + b
        m = (ii + 1) / (len(control_points) + 1)
        output[control_point] = m * delta + node_positions[source]
    return output


def _initialize_selfloops(edge_to_control_points, node_positions,
                          selfloop_radius = 0.1,
                          origin          = np.array([0, 0]),
                          scale           = np.array([1, 1])
):
    control_point_positions = dict()
    for (source, target), control_points in edge_to_control_points.items():
        # Source and target have the same position, such that
        # using the strategy employed above the control points
        # also end up at the same position. Instead we make a loop.
        control_point_positions.update(
            _init_selfloop(source, control_points, node_positions, selfloop_radius, origin, scale)
        )
    return control_point_positions


def _init_selfloop(source, control_points, node_positions, selfloop_radius, origin, scale):
    # To minimise overlap with other edges, we want the loop to be
    # on the side of the node away from the centroid of the graph.
    if len(node_positions) > 1:
        centroid = np.mean(list(node_positions.values()), axis=0)
        delta = node_positions[source] - centroid
        distance = np.linalg.norm(delta)
        unit_vector = delta / distance
    else: # single node in graph; self-loop points upwards
        unit_vector = np.array([0, 1])

    selfloop_center = node_positions[source] + selfloop_radius * unit_vector

    selfloop_control_point_positions = _get_n_points_on_a_circle(
        selfloop_center, selfloop_radius, len(control_points),
        _get_signed_angle_between(np.array([1., 0.]), node_positions[source] - selfloop_center)
    )

    # ensure that the loop stays within the bounding box
    selfloop_control_point_positions = _clip_to_frame(selfloop_control_point_positions, origin, scale)

    output = dict()
    for ii, control_point in enumerate(control_points):
        output[control_point] = selfloop_control_point_positions[ii]

    return output


def _optimize_control_point_positions(
        expanded_edge_list, node_positions,
        control_point_positions, total_control_points_per_edge,
        origin                        = np.array([0, 0]),
        scale                         = np.array([1, 1]),
        k                             = None,
        initial_temperature           = 0.1,
        total_iterations              = 50,
        node_size                     = None,
):

    # If the spacing of nodes is approximately k, the spacing of control points should be k / (total control points per edge + 1).
    # This would maximise the use of the available space. However, we do not want space to be filled with edges like a Peano-curve.
    # Therefor, we apply an additional fudge factor that pulls the edges a bit more taut.
    unique_nodes = list(node_positions.keys())
    if k is None:
        total_nodes = len(unique_nodes)
        area = np.product(scale)
        k = np.sqrt(area / float(total_nodes)) / (total_control_points_per_edge + 1)
        k *= 0.5

    control_point_positions.update(node_positions)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)

        expanded_node_positions = get_fruchterman_reingold_layout(
            expanded_edge_list,
            node_positions      = control_point_positions,
            scale               = scale,
            origin              = origin,
            k                   = k,
            initial_temperature = initial_temperature,
            total_iterations    = total_iterations,
            node_size           = node_size,
            fixed_nodes         = unique_nodes,
        )

    return expanded_node_positions


def _get_path_through_control_points(edge_to_control_points, expanded_node_positions):
    edge_to_path = dict()
    for (source, target), control_points in edge_to_control_points.items():
        control_point_positions = [expanded_node_positions[source]] \
            + [expanded_node_positions[node] for node in control_points] \
            + [expanded_node_positions[target]]
        edge_to_path[(source, target)] = control_point_positions
    return edge_to_path


def _fit_splines_through_control_points(edge_to_control_points, expanded_node_positions):
    # Fit a BSpline to each set of control points (+ anchors).
    edge_to_path = dict()
    for (source, target), control_points in edge_to_control_points.items():
        control_point_positions = [expanded_node_positions[source]] \
            + [expanded_node_positions[node] for node in control_points] \
            + [expanded_node_positions[target]]
        path = _bspline(np.array(control_point_positions))
        edge_to_path[(source, target)] = path
    return edge_to_path


@profile
def get_bundled_edge_paths(edge_list, node_positions,
                           k                       = 1000.,
                           compatibility_threshold = 0.05,
                           total_cycles            = 5,
                           total_iterations        = 50,
                           step_size               = 0.04,
):
    """Bundle edges using the FDEB algorithm proposed by Holten & Wijk (2009).

    This implementation follows the paper closely with the exception
    that instead of doubling the number of control point on each
    iteration (2n), a new control point is inserted between each
    existing pair of control points (2(n-1)+1), as proposed e.g. in Wu
    et al. (2015).

    Arguments:
    ----------
    edge_list : list
        List of (source node, target node) 2-tuples.

    node_positions : dict node : (float x, float y)
        Dictionary mapping nodes to (x, y) positions.

    k : float (default 1000.)
        The stiffness of the springs that connect control points.

    compatibility_threshold : float [0, 1] (default 0.05)
        Edge pairs with a lower compatibility score are not bundled together.
        Set to zero to bundle all edges with each other regardless of compatibility.

    total_cycles : int (default 5)
        The number of cycles. The number of control points (P) is doubled each cycle.

    total_iterations : int (default 50)
        Number of iterations (I) in the first cycle. Iterations are reduced by 1/3 with each cycle.

    step_size : float (default 0.04)
        Maximum step size (S) in the first cycle. Step sizes are halved each cycle.

    Returns:
    --------
    edge_to_paths : dict edge : path
        Dictionary mapping edges to arrays of (x, y) tuples, the edge segments.

    """

    # Filter out self-loops.
    if np.any([source == target for source, target in edge_list]):
        warnings.warn('Edge-bundling of self-loops not supported. Self-loops are removed from the edge list.')
        edge_list = [(source, target) for (source, target) in edge_list if source != target]

    # Filter out bi-directional edges.
    unidirectional_edges = set()
    for (source, target) in edge_list:
        if (target, source) not in unidirectional_edges:
            unidirectional_edges.add((source, target))
    reverse_edges = list(set(edge_list) - unidirectional_edges)
    edge_list = list(unidirectional_edges)

    edge_to_k = _get_k(edge_list, node_positions, k)

    edge_compatibility = _get_edge_compatibility(edge_list, node_positions, compatibility_threshold)

    edge_to_control_points = _initialize_control_points(edge_list, node_positions)

    for _ in range(total_cycles):
        edge_to_control_points = _expand_control_points(edge_to_control_points)

        for _ in range(total_iterations):
            F = _get_Fs(edge_to_control_points, edge_to_k)
            F = _get_Fe(edge_to_control_points, edge_compatibility, F)
            edge_to_control_points = _update_control_point_positions(
                edge_to_control_points, F, step_size)

        step_size /= 2.
        total_iterations = int(2/3 * total_iterations)

    # Add previously removed bi-directional edges back in.
    for (source, target) in reverse_edges:
        edge_to_control_points[(source, target)] = edge_to_control_points[(target, source)]

    return edge_to_control_points


def _get_k(edge_list, node_positions, k):
    return {(s, t) : k / np.linalg.norm(node_positions[t] - node_positions[s]) for (s, t) in edge_list}


@profile
def _get_edge_compatibility(edge_list, node_positions, threshold):
    # precompute edge segments, segment lengths and corresponding vectors
    edge_to_segment = {edge : Segment(node_positions[edge[0]], node_positions[edge[1]]) for edge in edge_list}

    edge_compatibility = list()
    for e1, e2 in itertools.combinations(edge_list, 2):
        P = edge_to_segment[e1]
        Q = edge_to_segment[e2]

        compatibility = 1
        compatibility *= _get_scale_compatibility(P, Q)
        if compatibility < threshold:
            continue # with next edge pair
        compatibility *= _get_position_compatibility(P, Q)
        if compatibility < threshold:
            continue # with next edge pair
        compatibility *= _get_angle_compatibility(P, Q)
        if compatibility < threshold:
            continue # with next edge pair
        compatibility *= _get_visibility_compatibility(P, Q)
        if compatibility < threshold:
            continue # with next edge pair

        # Also determine if one of the edges needs to be reversed:
        reverse = min(np.linalg.norm(P[0] - Q[0]), np.linalg.norm(P[1] - Q[1])) > \
            min(np.linalg.norm(P[0] - Q[1]), np.linalg.norm(P[1] - Q[0]))

        edge_compatibility.append((e1, e2, compatibility, reverse))

    return edge_compatibility


class Segment(object):
    def __init__(self, p0, p1):
        self.p0 = p0
        self.p1 = p1
        self.vector = p1 - p0
        self.length = np.linalg.norm(self.vector)
        self.unit_vector = self.vector / self.length
        self.midpoint = self.p0 * 0.5 * self.vector

    def __getitem__(self, idx):
        if idx == 0:
            return self.p0
        elif (idx == 1) or (idx == -1):
            return self.p1
        else:
            raise IndexError

    def get_orthogonal_projection_onto_segment(self, point):
        # Adapted from https://stackoverflow.com/a/61343727/2912349
        # The line extending the segment is parameterized as p0 + t (p1 - p0).
        # The projection falls where t = [(point-p0) . (p1-p0)] / |p1-p0|^2
        t = np.sum((point - self.p0) * self.vector) / self.length**2
        return self.p0 + t * self.vector

#     def get_interior_angle_with(self, other_segment):
#         # Adapted from: https://stackoverflow.com/a/13849249/2912349
#         return np.arccos(np.clip(np.dot(self.unit_vector, other_segment.unit_vector), -1.0, 1.0))


# def _get_angle_compatibility(P, Q):
#     return np.abs(np.cos(P.get_interior_angle_with(Q)))


def _get_angle_compatibility(P, Q):
    return np.abs(np.clip(np.dot(P.unit_vector, Q.unit_vector), -1.0, 1.0))


def _get_scale_compatibility(P, Q):
    avg = 0.5 * (P.length + Q.length)

    # The definition in the paper is rubbish, as the result is not on the interval [0, 1].
    # For example, consider an two edges, both 0.5 long:
    # return 2 / (avg * min(length_P, length_Q) + max(length_P, length_Q) / avg)
    # return min(length_P/length_Q, length_Q/length_P)
    return 2 / (avg / min(P.length, Q.length) + max(P.length, Q.length) / avg)


def _get_position_compatibility(P, Q):
    avg = 0.5 * (P.length + Q.length)
    distance_between_midpoints = np.linalg.norm(Q.midpoint - P.midpoint)
    return avg / (avg + distance_between_midpoints)


def _get_visibility_compatibility(P, Q):
    return min(_get_visibility(P, Q), _get_visibility(Q, P))


@profile
def _get_visibility(P, Q):
    I0 = P.get_orthogonal_projection_onto_segment(Q[0])
    I1 = P.get_orthogonal_projection_onto_segment(Q[1])
    I = Segment(I0, I1)
    distance_between_midpoints = np.linalg.norm(P.midpoint - I.midpoint)
    visibility = 1 - 2 * distance_between_midpoints / I.length
    return max(visibility, 0)


def _initialize_control_points(edge_list, node_positions):
    edge_to_control_points = dict()
    for source, target in edge_list:
        edge_to_control_points[(source, target)] \
            = np.array([node_positions[source], node_positions[target]])
    return edge_to_control_points


def _expand_control_points(edge_to_control_points):
    "Place a new control point between each pair of existing control points."
    for edge, control_points_old in edge_to_control_points.items():
        total_control_points_old = len(control_points_old)
        total_control_points_new = 2 * (total_control_points_old - 1) + 1
        control_points_new = np.zeros((total_control_points_new, 2))
        for ii in range(total_control_points_new):
            if (ii+1) % 2: # ii is even
                control_points_new[ii] = control_points_old[int(ii/2)]
            else: # ii is odd
                p1 = control_points_old[int((ii-1)/2)]
                p2 = control_points_old[int((ii+1)/2)]
                control_points_new[ii] = 0.5 * (p2 - p1) + p1
        edge_to_control_points[edge] = control_points_new
    return edge_to_control_points


def _get_Fs(edge_to_control_points, k):
    out = dict()
    for edge, control_points in edge_to_control_points.items():
        delta = np.zeros_like(control_points)
        diff = np.diff(control_points, axis=0)
        delta[1:-1] -= diff[:-1]
        delta[1:-1] += diff[1:]
        kp = k[edge] / (len(control_points) - 1)
        out[edge] = kp * delta
    return out


@profile
def _get_Fe(edge_to_control_points, edge_compatibility, out):

    for e1, e2, compatibility, reverse in edge_compatibility:
        P = edge_to_control_points[e1]
        Q = edge_to_control_points[e2]

        if not reverse:
            # i.e. if source/source or target/target closest
            delta = Q - P
        else:
            # need to reverse one set of control points
            delta = Q[::-1] - P

        # # desired computation:
        # distance = np.linalg.norm(delta, axis=1)
        # displacement = compatibility * delta / distance[..., None]**2

        # actually much faster:
        distance_squared = delta[:, 0]**2 + delta[:, 1]**2
        displacement = compatibility * delta / distance_squared[..., None]

        # Don't move the first and last control point, which are just the node positions.
        displacement[0] = 0
        displacement[-1] = 0

        out[e1] += displacement
        if not reverse:
            out[e2] -= displacement
        else:
            out[e2] -= displacement[::-1]

    return out


def _update_control_point_positions(edge_to_control_points, F, step_size):
    for edge, displacement in F.items():
        displacement_length = np.clip(np.linalg.norm(displacement), 1e-12, None) # prevent divide by 0 error in next line
        displacement = displacement / displacement_length * np.clip(displacement_length, None, step_size)
        edge_to_control_points[edge] += displacement
    return edge_to_control_points
