#!/usr/bin/env python
# coding: utf-8

"""
Edge routing routines.
"""

import itertools
import warnings
import numpy as np

from uuid import uuid4
from functools import wraps
from scipy.interpolate import UnivariateSpline
from matplotlib.patches import ConnectionStyle

from ._utils import (
    _bspline,
    _get_n_points_on_a_circle,
    _get_angle,
    _get_unit_vector,
    _edge_list_to_adjacency_list,
    _get_connected_components,
)


# monkeypatch repulsive force
def _get_fr_repulsion(distance, direction, k):
    """Compute repulsive forces.

    This is a variant of the implementation in the original FR
    algorithm, in as much as repulsion only acts between fixed nodes
    and mobile nodes, not between fixed nodes and other fixed nodes.
    """
    total_mobile = distance.shape[1]
    distance = distance[total_mobile:]
    direction = direction[total_mobile:]
    magnitude = k**2 / distance
    vectors = direction * magnitude[..., None]
    return np.sum(vectors, axis=0)

import netgraph._node_layout
netgraph._node_layout._get_fr_repulsion = _get_fr_repulsion
from netgraph._node_layout import get_fruchterman_reingold_layout, _clip_to_frame


# for profiling with kernprof/line_profiler
try:
    profile
except NameError:
    profile = lambda x: x


def _handle_multiple_components(layout_function):
    """If the graph contains multiple components, apply the given layout to each component individually."""
    @wraps(layout_function)
    def wrapped_layout_function(edges, node_positions=None, *args, **kwargs):
        adjacency_list = _edge_list_to_adjacency_list(edges, directed=False)
        components = _get_connected_components(adjacency_list)

        if len(components) > 1:
            return _get_layout_for_multiple_components(edges, node_positions, components, layout_function, *args, **kwargs)
        else:
            return layout_function(edges, node_positions, *args, **kwargs)

    return wrapped_layout_function


def _get_layout_for_multiple_components(edges, node_positions, components, layout_function, *args, **kwargs):
    """Partition network into given components and apply the given layout to each component individually."""
    edge_paths = dict()
    for component in components:
        component_edges = [(source, target) for (source, target) in edges if (source in component) and (target in component)]
        component_node_positions = {node : xy for node, xy in node_positions.items() if node in component}
        component_edge_paths = layout_function(component_edges, component_node_positions, *args, **kwargs)
        edge_paths.update(component_edge_paths)
    return edge_paths


def get_straight_edge_paths(edges, node_positions, edge_width):
    """Edge routing using straight lines.

    Computes the edge paths, such that edges are represented by
    straight lines connecting the source and target node.
    Bi-directional edges are offset from one another by one edge width.

    Parameters
    ----------
    edges : list
        The edges of the graph, with each edge being represented by a (source node ID, target node ID) tuple.
    node_positions : dict
        Dictionary mapping each node ID to (float x, float y) tuple, the node position.
    edge_width: dict
        Dictionary mapping each edge to a float, the edge width.

    Returns
    -------
    edge_paths : dict
        Dictionary mapping each edge to an array of (x, y) coordinates representing its path.

    """
    edge_paths = dict()
    for (source, target) in edges:
        if source == target:
            # msg = "Plotting of self-loops not supported for straight edges."
            # msg += "Ignoring edge ({}, {}).".format(source, target)
            # warnings.warn(msg)
            continue

        x1, y1 = node_positions[source]
        x2, y2 = node_positions[target]

        # if (target, source) in edges: # i.e. bidirectional
        #     # shift edge to the right (looking along the arrow)
        #     x1, y1, x2, y2 = _shift_edge(x1, y1, x2, y2, delta=-0.5*edge_width[(source, target)])

        edge_paths[(source, target)] = np.c_[[x1, x2], [y1, y2]]

    return edge_paths


def _shift_edge(x1, y1, x2, y2, delta):
    """Determine the parallel to a segment defined by points p1: (x1, y1) and p2 : (x2, y2) at a distance delta."""
    # convert segment into a vector
    v = np.r_[x2-x1, y2-y1]
    # compute orthogonal vector
    v = np.r_[-v[1], v[0]]
    # convert to orthogonal unit vector
    v = v / np.linalg.norm(v)
    # compute offsets
    dx, dy = delta * v
    # return new coordinates of point p1' and p2'
    return x1+dx, y1+dy, x2+dx, y2+dy


def get_selfloop_paths(edges, node_positions, selfloop_radius, origin, scale, angle=None):
    """Edge routing for self-loops.

    Parameters
    ----------
    edges : list
        The edges of the graph, with each edge being represented by a (source node ID, target node ID) tuple.
    node_positions : dict
        Dictionary mapping each node ID to (float x, float y) tuple, the node position.
    selfloop_radius : float
        The radius of the self-loops.
    origin : numpy.array
        A (float x, float y) tuple corresponding to the lower left hand corner of the bounding box specifying the extent of the canvas.
    scale : numpy.array
        A (float x, float y) tuple representing the width and height of the bounding box specifying the extent of the canvas.
    angle : float or None, default None
        The starting angle of the self-loop in radians. If None, the self-loop is drawn opposite of the centroid of the graph.

    Returns
    -------
    edge_paths : dict
        Dictionary mapping each edge to an array of (x, y) coordinates representing its path.

    """

    edge_paths = dict()
    for (source, target) in edges:
        if source != target:
            # msg = "Edges must be self-loops."
            # msg += f"Ignoring edge ({source}, {target})."
            # warnings.warn(msg)
            continue

        edge_paths[(source, target)] = _get_selfloop_path(
            source, node_positions, selfloop_radius, origin, scale, angle)

    return edge_paths


def _get_selfloop_path(source, node_positions, selfloop_radius, origin, scale, angle=None):
    """Compute the edge path for a single self-loop."""

    x, y = node_positions[source]

    if angle is not None:
        unit_vector = _get_unit_vector(np.array([np.cos(angle), np.sin(angle)]))
    else:
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

    selfloop_path = _get_n_points_on_a_circle(
        selfloop_center, selfloop_radius, 100+1,
        _get_angle(*unit_vector) + np.pi,
    )[1:]

    # ensure that the loop stays within the bounding box
    selfloop_path = _clip_to_frame(selfloop_path, origin, scale)

    return selfloop_path


def get_curved_edge_paths(edges, node_positions,
                          selfloop_radius               = 0.1,
                          origin                        = np.array([0, 0]),
                          scale                         = np.array([1, 1]),
                          k                             = None,
                          initial_temperature           = 0.01,
                          total_iterations              = 50,
                          node_size                     = 0.):
    """Edge routing using curved paths that avoid nodes and other edges.

    Computes the edge paths, such that edges are represented by curved
    lines connecting the source and target node. Edges paths avoid
    nodes and each other. The edge layout is determined using the
    Fruchterman-Reingold algorithm.

    Parameters
    ----------
    edges : list
        The edges of the graph, with each edge being represented by a (source node ID, target node ID) tuple.
    node_positions : dict
        Dictionary mapping each node ID to (float x, float y) tuple, the node position.
    selfloop_radius : float
        The radius of the self-loops.
    origin : numpy.array
        A (float x, float y) tuple corresponding to the lower left hand corner of the bounding box specifying the extent of the canvas.
    scale : numpy.array
        A (float x, float y) tuple representing the width and height of the bounding box specifying the extent of the canvas.
    k : float or None, default None
        Spring constant, which controls the tautness of edges.
        Small values will result in straight connections, large values in bulging arcs.
        If None, initialized to: 0.1 * sqrt(area / total nodes).
    total_iterations : int, default 50
        Number of iterations in the Fruchterman-Reingold algorithm.
    initial_temperature: float, default 1.
        Temperature controls the maximum node displacement on each iteration.
        Temperature is decreased on each iteration to eventually force the algorithm
        into a particular solution. The size of the initial temperature determines how
        quickly that happens. Values should be much smaller than the values of `scale`.
    node_size : dict
        Dictionary mapping each node to a float, the node size. Used for node avoidance.

    Returns
    -------
    edge_paths : dict
        Dictionary mapping each edge to an array of (x, y) coordinates representing its path.

    """

    # If the spacing of nodes is approximately k, the spacing of control points should be k / (total control points per edge + 1).
    # This would maximise the use of the available space. However, we do not want space to be filled with edges like a Peano-curve.
    # Therefor, we apply an additional fudge factor that pulls the edges a bit more taut.
    if k is None:
        total_nodes = len(nodes)
        area = np.product(scale)
        k = np.sqrt(area / float(total_nodes))
        k *= 0.1

    edge_to_control_points = _initialize_control_points(edges, node_positions, k, scale)

    control_point_positions = _initialize_control_point_positions(
        edge_to_control_points, node_positions, selfloop_radius, origin, scale)

    control_point_positions = _optimize_control_point_positions(
        edge_to_control_points, node_positions, control_point_positions,
        origin, scale, k, initial_temperature, total_iterations, node_size,
    )

    edge_to_path = _get_path_through_control_points(
        edge_to_control_points, node_positions, control_point_positions)

    edge_to_path = _fit_splines_through_edge_paths(edge_to_path)

    return edge_to_path


def _initialize_control_points(edges, node_positions, k, scale):
    """Represent each edge with string of control points."""
    edge_to_control_points = dict()
    for start, stop in edges:
        if start != stop:
            distance = np.linalg.norm(node_positions[stop] - node_positions[start], axis=-1) / np.linalg.norm(scale)
            # total_control_points = distance * np.pi / k # approximating the arc length with a half-circle
            total_control_points = distance * 10
            total_control_points = min(max(int(total_control_points), 1), 5) # ensure that there are at least one point but no more than 5
            edge_to_control_points[(start, stop)] = [uuid4() for _ in range(total_control_points)]
        else: # self-loop
            edge_to_control_points[(start, stop)] = [uuid4() for _ in range(5)]
    return edge_to_control_points


def _expand_edges(edge_to_control_points):
    """Create a new, expanded edge list, in which each edge is split into multiple segments.
    There are total_control_points + 1 segments / edges for each original edge.

    """
    expanded_edges = []
    for (source, target), control_points in edge_to_control_points.items():
        sources = [source] + control_points
        targets = control_points + [target]
        expanded_edges.extend(zip(sources, targets))
    return expanded_edges


def _initialize_control_point_positions(edge_to_control_points, node_positions,
                                        selfloop_radius = 0.1,
                                        origin          = np.array([0, 0]),
                                        scale           = np.array([1, 1])):
    """Initialise the positions of the control points to positions on a straight
    line between source and target node. For self-loops, initialise the positions
    on a circle next to the node.

    """

    nonloops_to_control_points = {(source, target) : pts for (source, target), pts in edge_to_control_points.items() if source != target}
    selfloops_to_control_points = {(source, target) : pts for (source, target), pts in edge_to_control_points.items() if source == target}

    control_point_positions = dict()
    control_point_positions.update(_initialize_nonloops(nonloops_to_control_points, node_positions))
    control_point_positions.update(_initialize_selfloops(selfloops_to_control_points, node_positions, selfloop_radius, origin, scale))

    return control_point_positions


def _initialize_nonloops(edge_to_control_points, node_positions):
    """Merge control point : position dictionary for different non self-loops into a single dictionary."""
    control_point_positions = dict()
    for (source, target), control_points in edge_to_control_points.items():
        control_point_positions.update(_init_nonloop(source, target, control_points, node_positions))
    return control_point_positions


def _init_nonloop(source, target, control_points, node_positions):
    """Initialise the positions of the control points to positions on a straight line between source and target node."""
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
                          scale           = np.array([1, 1])):
    """Merge control point : position dictionary for different self-loops into a single dictionary."""
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
    """Initialise the positions of control points to positions on a circle next to the node."""
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
        selfloop_center, selfloop_radius, len(control_points)+1,
        _get_angle(*unit_vector) + np.pi,
    )[1:]

    # ensure that the loop stays within the bounding box
    selfloop_control_point_positions = _clip_to_frame(selfloop_control_point_positions, origin, scale)

    output = dict()
    for ii, control_point in enumerate(control_points):
        output[control_point] = selfloop_control_point_positions[ii]

    return output


def _optimize_control_point_positions(
        edge_to_control_points, node_positions, control_point_positions,
        origin, scale, k, initial_temperature, total_iterations, node_size):
    """Optimise the position of control points using the FR algorithm."""
    nodes = list(node_positions.keys())
    expanded_edges = _expand_edges(edge_to_control_points)
    expanded_node_positions = control_point_positions.copy() # TODO: may need deepcopy here
    expanded_node_positions.update(node_positions)

    # increase size of nodes so that there is a bit more clearance between edges and nodes
    node_size = {node : 2 * size for node, size in node_size.items()}

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)

        expanded_node_positions = get_fruchterman_reingold_layout.__wrapped__(
            expanded_edges,
            node_positions      = expanded_node_positions,
            scale               = scale,
            origin              = origin,
            k                   = k,
            initial_temperature = initial_temperature,
            total_iterations    = total_iterations,
            node_size           = node_size,
            fixed_nodes         = nodes,
        )

    return {node : xy for node, xy in expanded_node_positions.items() if node not in nodes}


def _get_path_through_control_points(edge_to_control_points, node_positions, control_point_positions):
    """Map each edge to an array of (optimised) control points positions."""
    edge_to_path = dict()
    for (source, target), control_points in edge_to_control_points.items():
        path = [node_positions[source]] \
            + [control_point_positions[node] for node in control_points] \
            + [node_positions[target]]
        edge_to_path[(source, target)] = np.array(path)
    return edge_to_path


def _fit_splines_through_edge_paths(edge_to_path, *args, **kwargs):
    """Fit splines through edge paths for smoother edge routing."""
    return {edge : _bspline(path, *args, **kwargs) for edge, path in edge_to_path.items()}


@_handle_multiple_components
def get_arced_edge_paths(edges, node_positions, rad=1.,
                         origin = np.array([0, 0]),
                         scale  = np.array([1, 1]),
):

    """Determine the edge layout, where edges are represented by arcs
    connecting the source and target node.

    Creates simple quadratic Bezier curves between nodes. The curves
    are created so that the middle control points (C1) are located at
    the same distance from the start (C0) and end points (C2) and the
    distance of the C1 to the line connecting C0-C2 is rad times the
    distance of C0-C2.

    Arguments:
    ----------
    edges : list of (source node ID, target node ID) 2-tuples
        The edges.

    node_positions : dict node ID : (x, y) positions
        The node positions.

    rad : float (default 1.0)
        The curvature of the arc.

    origin : (float x, float y) tuple or None (default (0, 0))
        The lower left hand corner of the bounding box specifying the extent of the layout.

    scale : (float delta x, float delta y) or None (default (1, 1))
        The width and height of the bounding box specifying the extent of the layout.

    Returns:
    --------
    edge_paths : dict edge : ndarray
        Dictionary mapping each edge to a list of edge segments.

    """
    # TODO: ensure that arcs are within bbox given by origin and scale
    edge_paths = dict()
    for source, target in edges:
        if source == target:
            # msg = "Plotting of self-loops not supported for straight edges."
            # msg += "Ignoring edge ({}, {}).".format(source, target)
            # warnings.warn(msg)
            continue
        arc_factory = ConnectionStyle.Arc3(rad=rad)
        path = arc_factory(
            node_positions[source],
            node_positions[target],
            shrinkA=0., shrinkB=0.
            )
        edge_paths[(source, target)] = _bspline(path.vertices, 100)

    return edge_paths


@profile
@_handle_multiple_components
def get_bundled_edge_paths(edges, node_positions,
                           k                       = 1000.,
                           compatibility_threshold = 0.05,
                           total_cycles            = 5,
                           total_iterations        = 50,
                           step_size               = 0.04,
                           straighten_by           = 0.,
):
    """Edge routing with bundled edge paths.

    Uses the FDEB algorithm as proposed in [Holten2009]_.
    This implementation follows the paper closely with the exception
    that instead of doubling the number of control point on each
    iteration (2n), a new control point is inserted between each
    existing pair of control points (2(n-1)+1), as proposed e.g. in Wu
    et al. (2015) [Wu2015]_.

    Parameters
    ----------

    edges : list
        The edges of the graph, with each edge being represented by a (source node ID, target node ID) tuple.
    node_positions : dict
        Dictionary mapping each node ID to (float x, float y) tuple, the node position.
    k : float, default 1000.
        The stiffness of the springs that connect control points.
    compatibility_threshold : float, default 0.05
        Edge pairs with a lower compatibility score are not bundled together.
        Set to zero to bundle all edges with each other regardless of compatibility.
        Set to one to prevent bundling of any (non-identical) edges.
    total_cycles : int, default 5
        The number of cycles. The number of control points (P) is doubled each cycle.
    total_iterations : int, default 50
        Number of iterations (I) in the first cycle. Iterations are reduced by 1/3 with each cycle.
    step_size : float, default 0.04
        Maximum step size (S) in the first cycle. Step sizes are halved each cycle.
    straighten_by : float, default 0.
        The amount of edge straightening applied after bundling.
        A small amount of straightening can help indicating the number of
        edges comprising a bundle by widening the bundle.
        If set to one, edges are fully un-bundled and plotted as stright lines.

    Returns
    -------
    edge_paths : dict
        Dictionary mapping each edge to an array of (x, y) coordinates representing its path.

    References
    ----------
    .. [Holten2009] Holten D and Van Wijk JJ. (2009) ‘Force-Directed edge
       bundling for graph visualization’, Computer Graphics Forum.

    .. [Wu2015] Wu J, Yu L, Yu H (2015) ‘Texture-based edge bundling: A
       web-based approach for interactively visualizing large graphs’,
       IEEE International Conference on Big Data.

    """

    # Filter out self-loops.
    if np.any([source == target for source, target in edges]):
        warnings.warn('Edge-bundling of self-loops not supported. Self-loops are removed from the edge list.')
        edges = [(source, target) for (source, target) in edges if source != target]

    # Filter out bi-directional edges.
    unidirectional_edges = set()
    for (source, target) in edges:
        if (target, source) not in unidirectional_edges:
            unidirectional_edges.add((source, target))
    reverse_edges = list(set(edges) - unidirectional_edges)
    edges = list(unidirectional_edges)

    edge_to_k = _get_k(edges, node_positions, k)

    edge_compatibility = _get_edge_compatibility(edges, node_positions, compatibility_threshold)

    edge_to_control_points = _initialize_bundled_control_points(edges, node_positions)

    for _ in range(total_cycles):
        edge_to_control_points = _expand_control_points(edge_to_control_points)

        for _ in range(total_iterations):
            F = _get_Fs(edge_to_control_points, edge_to_k)
            F = _get_Fe(edge_to_control_points, edge_compatibility, F)
            edge_to_control_points = _update_control_point_positions(
                edge_to_control_points, F, step_size)

        step_size /= 2.
        total_iterations = int(2/3 * total_iterations)

    if straighten_by > 0.:
        edge_to_control_points = _straighten_edges(edge_to_control_points, straighten_by)

    edge_to_control_points = _smooth_edges(edge_to_control_points)

    # Add previously removed bi-directional edges back in.
    for (source, target) in reverse_edges:
        edge_to_control_points[(source, target)] = edge_to_control_points[(target, source)][::-1]

    return edge_to_control_points


def _get_k(edges, node_positions, k):
    """Assign each edge a stiffness depending on its length and the global stiffness constant."""
    return {(s, t) : k / np.linalg.norm(node_positions[t] - node_positions[s]) for (s, t) in edges}


@profile
def _get_edge_compatibility(edges, node_positions, threshold):
    """Compute the compatibility between all edge pairs."""
    # precompute edge segments, segment lengths and corresponding vectors
    edge_to_segment = {edge : Segment(node_positions[edge[0]], node_positions[edge[1]]) for edge in edges}

    edge_compatibility = list()
    for e1, e2 in itertools.combinations(edges, 2):
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
    """Compute the angle compatibility between two segments P and Q.
    The angle compatibility is high if the interior angle between them is small.

    """
    return np.abs(np.clip(np.dot(P.unit_vector, Q.unit_vector), -1.0, 1.0))


def _get_scale_compatibility(P, Q):
    """Compute the scale compatibility between two segments P and Q.
    The scale compatibility is high if their lengths are similar.

    """
    avg = 0.5 * (P.length + Q.length)

    # The definition in the paper is rubbish, as the result is not on the interval [0, 1].
    # For example, consider an two edges, both 0.5 long:
    # return 2 / (avg * min(length_P, length_Q) + max(length_P, length_Q) / avg)

    # my original alternative:
    # return min(length_P/length_Q, length_Q/length_P)

    # typo in original paper corrected in Graser et al. (2019)
    return 2 / (avg / min(P.length, Q.length) + max(P.length, Q.length) / avg)


def _get_position_compatibility(P, Q):
    """Compute the position compatibility between two segments P and Q.
    The position compatibility is high if the distance between their midpoints is small.

    """
    avg = 0.5 * (P.length + Q.length)
    distance_between_midpoints = np.linalg.norm(Q.midpoint - P.midpoint)
    # This is the definition from the paper, but the scaling should probably be more aggressive.
    return avg / (avg + distance_between_midpoints)


def _get_visibility_compatibility(P, Q):
    """Compute the visibility compatibility between two segments P and Q.
    The visibility compatibility is low if bundling would occlude any of the end points of the segments.

    """
    return min(_get_visibility(P, Q), _get_visibility(Q, P))


@profile
def _get_visibility(P, Q):
    I0 = P.get_orthogonal_projection_onto_segment(Q[0])
    I1 = P.get_orthogonal_projection_onto_segment(Q[1])
    I = Segment(I0, I1)
    distance_between_midpoints = np.linalg.norm(P.midpoint - I.midpoint)
    visibility = 1 - 2 * distance_between_midpoints / I.length
    return max(visibility, 0)


def _initialize_bundled_control_points(edges, node_positions):
    """Initialise each edge with two control points, the positions of the source and target nodes."""
    edge_to_control_points = dict()
    for source, target in edges:
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
    """Compute all spring forces."""
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
    """Compute all electrostatic forces."""
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
    """Update control point positions using the calculated net forces."""
    for edge, displacement in F.items():
        displacement_length = np.clip(np.linalg.norm(displacement), 1e-12, None) # prevent divide by 0 error in next line
        displacement = displacement / displacement_length * np.clip(displacement_length, None, step_size)
        edge_to_control_points[edge] += displacement
    return edge_to_control_points


def _smooth_edges(edge_to_path):
    """Wraps _smooth_path()."""
    return {edge : _smooth_path(path) for edge, path in edge_to_path.items()}


def _smooth_path(path):
    """Smooth a path by fitting a univariate spline.

    Notes
    -----
    Adapted from https://stackoverflow.com/a/52020098/2912349

    """

    # Compute the linear length along the line:
    distance = np.cumsum( np.sqrt(np.sum( np.diff(path, axis=0)**2, axis=1 )) )
    distance = np.insert(distance, 0, 0)/distance[-1]

    # Compute a spline function for each dimension:
    splines = [UnivariateSpline(distance, coords, k=3, s=.001) for coords in path.T]

    # Computed the smoothed path:
    alpha = np.linspace(0, 1, 100)
    return np.vstack([spl(alpha) for spl in splines]).T


def _straighten_edges(edge_to_path, straighten_by):
    """Wraps _straigthen_path()"""
    return {edge : _straighten_path(path, straighten_by) for edge, path in edge_to_path.items()}


def _straighten_path(path, straighten_by):
    """Straigthen a path by computing the weighted average between the path and
    a straight line connecting the end points.

    """
    p0 = path[0]
    p1 = path[-1]
    n = len(path)
    return (1 - straighten_by) * path \
        + straighten_by * (p0 + np.linspace(0, 1, n)[:, np.newaxis] * (p1 - p0))
