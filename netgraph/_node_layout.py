#!/usr/bin/env python
# coding: utf-8

"""
Node layout routines.
"""

import warnings
import itertools
import numpy as np

from functools import wraps, partial
from itertools import combinations, product
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.optimize import minimize, NonlinearConstraint

from rpack import pack
from grandalf.graphs import Vertex, Edge, Graph
from grandalf.layouts import SugiyamaLayout, DummyVertex

from ._utils import (
    _edge_list_to_adjacency_matrix,
    _edge_list_to_adjacency_list,
    _get_subgraph,
    _get_unique_nodes,
    _get_n_points_on_a_circle,
    _invert_dict,
    _get_connected_components,
    _convert_polar_to_cartesian_coordinates,
    _get_angle,
)


DEBUG = False


def _handle_multiple_components(layout_function, packing_function):
    """Most layout algorithms only handle graphs that consist of a giant
    single component, and fail to find a suitable layout if the graph
    consists of more than component. This decorator wraps a given
    layout function such that if the graph contains more than one
    component, the layout is first computed for each individual
    component, and then the component layouts are combined using
    the packing function.

    """

    @wraps(layout_function)
    def wrapped_layout_function(edges, nodes=None, *args, **kwargs):

        # determine if there are more than one component
        adjacency_list = _edge_list_to_adjacency_list(edges, directed=False)
        components = _get_connected_components(adjacency_list)

        if nodes:
            unconnected_nodes = set(nodes) - set(_get_unique_nodes(edges))
            if unconnected_nodes:
                for node in unconnected_nodes:
                    components.append([node])

        if len(components) > 1:
            return get_layout_for_multiple_components(
                edges, components, layout_function, packing_function, *args, **kwargs)
        else:
            return layout_function(edges, *args, **kwargs)

    return wrapped_layout_function


def get_layout_for_multiple_components(edges, components,
                                       layout_function, packing_function,
                                       origin, scale, *args, **kwargs):
    """Determine suitable bounding box dimensions and placement for each
    component in the graph, and then compute the layout of each
    individual component given the constraint of the bounding box.

    Parameters
    ----------
    edges : list
        The edges of the graph, with each edge being represented by a (source node ID, target node ID) tuple.
    components : list of sets
        The connected components of the graph.
    layout_function : function
        The function used to compute the relative positions of each node within a component.
        The arguments and key-word arguments are passed through to this function.
    packing_function : function
        The function used to arrange component layouts w.r.t. each other.
    origin : tuple
        The (float x, float y) coordinates corresponding to the lower left corner of the bounding box specifying the extent of the canvas.
    scale : tuple
        The (float x, float y) dimensions representing the width and height of the bounding box specifying the extent of the canvas.

    *args, **kwargs
        Passed through to layout_function

    Returns
    -------
    node_positions : dict
        Dictionary mapping each node ID to (float x, float y) tuple, the node position.

    """

    bboxes = packing_function(components, origin, scale)

    node_positions = dict()
    for ii, (component, bbox) in enumerate(zip(components, bboxes)):
        if len(component) > 1:
            subgraph = _get_subgraph(edges, component)
            component_node_positions = layout_function(subgraph, origin=bbox[:2], scale=bbox[2:], *args, **kwargs)
            node_positions.update(component_node_positions)
        else:
            # component is a single node, which we can simply place at the centre of the bounding box
            node_positions[component.pop()] = np.array([bbox[0] + 0.5 * bbox[2], bbox[1] + 0.5 * bbox[3]])

    return node_positions


def _get_packed_component_bboxes(components, origin, scale, pad_by=0.1, power=0.8):
    """Partition the canvas given by origin and scale into bounding boxes, one for each component.

    Use rectangle packing to tightly arrange bounding boxes.

    Parameters
    ----------
    components : list of sets
        The connected components of the graph.
    origin : tuple
        The (float x, float y) coordinates corresponding to the lower left corner of the bounding box specifying the extent of the canvas.
    scale : tuple
        The (float x, float y) dimensions representing the width and height of the bounding box specifying the extent of the canvas.
    pad_by : float, default 0.05
        Padding around node positions to reduce clipping of the node artists with the frame,
        and to create space for routing curved edges including self-loops around nodes.
        This results in the following bounding box:

        .. code::

            xmin = origin[0] + pad_by * scale[0]
            ymin = origin[1] + pad_by * scale[1]
            xmax = origin[0] + scale[0] - pad_by * scale[0]
            ymax = origin[1] + scale[1] - pad_by * scale[1]

    power : float, default 0.8
        The dimensions each bounding box are given by |V|^power by |V|^power,
        where |V| are the total number of nodes.

    Returns
    -------
    bboxes : list of tuples
        The (left, bottom, width height) bounding boxes for each component.

    """

    relative_dimensions = [_get_bbox_dimensions(len(component), power=power) for component in components]

    # Add a padding between boxes, such that nodes cannot end up touching in the final layout.
    # We choose a padding proportional to the dimensions of the largest box.
    maximum_width, maximum_height = np.max(relative_dimensions, axis=0)
    pad_x, pad_y = pad_by * maximum_width, pad_by * maximum_height
    padded_dimensions = [(width + pad_x, height + pad_y) for (width, height) in relative_dimensions]

    # rpack only works on integers, hence multiply by some large scalar to retain some precision;
    # NB: for some strange reason, rpack's running time is sensitive to the size of the boxes, so don't make the scalar too large
    # TODO find alternative to rpack
    scalar = 1000 / ((1 + pad_by) * max(maximum_width, maximum_height))
    integer_dimensions = [(int(scalar*width), int(scalar*height)) for width, height in padded_dimensions]
    origins = pack(integer_dimensions) # NB: rpack claims to return upper-left corners, when it actually returns lower-left corners

    bboxes = [(x, y, scalar*width, scalar*height) for (x, y), (width, height) in zip(origins, relative_dimensions)]

    # rescale boxes to canvas, effectively reversing the upscaling
    bboxes = _rescale_bboxes_to_canvas(bboxes, origin, scale)

    if DEBUG:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        fig, ax = plt.subplots(1,1)
        for bbox in bboxes:
            ax.add_artist(Rectangle(bbox[:2], bbox[2], bbox[3], color=np.random.rand(3), zorder=-1))
        # plt.show()

    return bboxes


def _get_bbox_dimensions(n, power=0.5):
    """Given n nodes, compute appropriately scaled square bbox dimensions."""
    # TODO: factor in the dimensions of the canvas
    # such that the rescaled boxes are approximately square
    return (n**power, n**power)


def _rescale_bboxes_to_canvas(bboxes, origin, scale):
    """Convert relative bbox dimensions to absolute bbox dimensions given the dimensions of the available canvas."""
    lower_left_hand_corners = [(x, y) for (x, y, _, _) in bboxes]
    upper_right_hand_corners = [(x+w, y+h) for (x, y, w, h) in bboxes]
    minimum = np.min(lower_left_hand_corners, axis=0)
    maximum = np.max(upper_right_hand_corners, axis=0)
    total_width, total_height = maximum - minimum

    # shift to (0, 0)
    min_x, min_y = minimum
    lower_left_hand_corners = [(x - min_x, y-min_y) for (x, y) in lower_left_hand_corners]

    # rescale
    scale_x = scale[0] / total_width
    scale_y = scale[1] / total_height

    lower_left_hand_corners = [(x*scale_x, y*scale_y) for x, y in lower_left_hand_corners]
    dimensions = [(w * scale_x, h * scale_y) for (_, _, w, h) in bboxes]
    rescaled_bboxes = [(x, y, w, h) for (x, y), (w, h) in zip(lower_left_hand_corners, dimensions)]

    # shift by origin
    x0, y0 = origin
    shifted_bboxes = [(x + x0, y + y0, w, h) for (x, y, w, h) in rescaled_bboxes]
    return shifted_bboxes


def _get_side_by_side_component_bboxes(components, origin, scale, pad_by=0.05):
    """Partition the canvas given by origin and scale into bounding boxes, one for each component.

    Position bounding boxes next to each other.

    Parameters
    ----------
    components : list of sets
        The connected components of the graph.
    origin : tuple
        The (float x, float y) coordinates corresponding to the lower left corner of the bounding box specifying the extent of the canvas.
    scale : tuple
        The (float x, float y) dimensions representing the width and height of the bounding box specifying the extent of the canvas.

    Returns
    -------
    bboxes : list of tuples
        The (left, bottom, width height) bounding boxes for each component.

    """

    relative_dimensions = [(len(component), 1) for component in components]

    # Add a padding between boxes, such that nodes cannot end up touching in the final layout.
    # We choose a padding proportional to the dimensions of the largest box.
    maximum_dimensions = np.max(relative_dimensions, axis=0)
    pad_x, pad_y = pad_by * maximum_dimensions
    padded_dimensions = [(width + pad_x, height + pad_y) for (width, height) in relative_dimensions]

    x = 0
    origins = []
    for (dx, _) in padded_dimensions:
        origins.append((x, 0))
        x += dx

    bboxes = [(x, y, width, height) for (x, y), (width, height) in zip(origins, relative_dimensions)]

    # rescale boxes to canvas, effectively reversing the upscaling
    bboxes = _rescale_bboxes_to_canvas(bboxes, origin, scale)

    if DEBUG:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        fig, ax = plt.subplots(1,1)
        for bbox in bboxes:
            ax.add_artist(Rectangle(bbox[:2], bbox[2], bbox[3], color=np.random.rand(3)))
        plt.show()

    return bboxes


_rectangle_pack_multiple_components = \
    partial(_handle_multiple_components, packing_function=_get_packed_component_bboxes)

_arrange_multiple_components_side_by_side = \
    partial(_handle_multiple_components, packing_function=_get_side_by_side_component_bboxes)


def _get_fr_repulsion(distance, direction, k):
    """Compute repulsive forces."""
    with np.errstate(divide='ignore', invalid='ignore'):
        magnitude = k**2 / distance
    vectors = direction * magnitude[..., None]
    # Note that we cannot apply the usual strategy of summing the array
    # along either axis and subtracting the trace,
    # as the diagonal of `direction` is np.nan, and any sum or difference of
    # NaNs is just another NaN.
    # Also we do not want to ignore NaNs by using np.nansum, as then we would
    # potentially mask the existence of off-diagonal zero distances.
    for ii in range(vectors.shape[-1]):
        np.fill_diagonal(vectors[:, :, ii], 0)
    return np.sum(vectors, axis=0)


def _get_fr_attraction(distance, direction, adjacency, k):
    """Compute attractive forces."""
    magnitude = 1./k * distance**2 * adjacency
    vectors = -direction * magnitude[..., None] # NB: the minus!
    for ii in range(vectors.shape[-1]):
        np.fill_diagonal(vectors[:, :, ii], 0)
    return np.sum(vectors, axis=0)


@_rectangle_pack_multiple_components
def get_fruchterman_reingold_layout(edges,
                                    edge_weights        = None,
                                    k                   = None,
                                    origin              = (0, 0),
                                    scale               = (1, 1),
                                    pad_by              = 0.05,
                                    initial_temperature = 1.,
                                    total_iterations    = 50,
                                    node_size           = 0,
                                    node_positions      = None,
                                    fixed_nodes         = None,
                                    validate_positions  = True,
                                    get_repulsion       = _get_fr_repulsion,
                                    get_attraction      = _get_fr_attraction,
                                    *args, **kwargs):
    """'Spring' or Fruchterman-Reingold node layout.

    Uses the Fruchterman-Reingold algorithm [Fruchterman1991]_ to compute node positions.
    This algorithm simulates the graph as a physical system, in which nodes repell each other.
    For connected nodes, this repulsion is counteracted by an attractive force exerted by the edges, which are simulated as springs.
    The resulting layout is hence often referred to as a 'spring' layout.

    Parameters
    ----------
    edges : list
        The edges of the graph, with each edge being represented by a (source node ID, target node ID) tuple.
    edge_weights : dict
        Mapping of edges to edge weights.
    k : float or None, default None
        Expected mean edge length. If None, initialized to the sqrt(area / total nodes).
    origin : tuple, default (0, 0)
        The (float x, float y) coordinates corresponding to the lower left corner of the bounding box specifying the extent of the canvas.
    scale : tuple, default (1, 1)
        The (float x, float y) dimensions representing the width and height of the bounding box specifying the extent of the canvas.
    pad_by : float, default 0.05
        Padding around node positions to reduce clipping of the node artists with the frame,
        and to create space for routing curved edges including self-loops around nodes.
        This results in the following bounding box:

        .. code-block::

           xmin = origin[0] + pad_by * scale[0]
           ymin = origin[1] + pad_by * scale[1]
           xmax = origin[0] + scale[0] - pad_by * scale[0]
           ymax = origin[1] + scale[1] - pad_by * scale[1]

    total_iterations : int, default 50
        Number of iterations.
    initial_temperature: float, default 1.
        Temperature controls the maximum node displacement on each iteration.
        Temperature is decreased on each iteration to eventually force the algorithm into a particular solution.
        The size of the initial temperature determines how quickly that happens.
        Values should be much smaller than the values of `scale`.
    node_size : scalar or dict, default 0.
        Size (radius) of nodes.
        Providing the correct node size minimises the overlap of nodes in the graph,
        which can otherwise occur if there are many nodes, or if the nodes differ considerably in size.
    node_positions : dict or None, default None
        Mapping of nodes to their (initial) x,y positions. If None are given,
        nodes are initially placed randomly within the bounding box defined by `origin` and `scale`.
        If the graph has multiple components, explicit initial positions may result in a ValueError,
        if the initial positions fall outside of the area allocated to that specific component.
    fixed_nodes : list or None, default None
        Nodes to keep fixed at their initial positions.

    Returns
    -------
    node_positions : dict
        Dictionary mapping each node ID to (float x, float y) tuple, the node position.

    References
    ----------
    .. [Fruchterman1991] Fruchterman, TMJ and Reingold, EM (1991) ‘Graph drawing by force‐directed placement’,
       Software: Practice and Experience

    """

    assert len(edges) > 0, "The list of edges has to be non-empty."

    # This is just a wrapper around `_fruchterman_reingold`, which implements (the loop body of) the algorithm proper.
    # This wrapper handles the initialization of variables to their defaults (if not explicitely provided),
    # and checks inputs for self-consistency.

    origin = np.array(origin)
    scale = np.array(scale)
    assert len(origin) == len(scale), \
        "Arguments `origin` (d={}) and `scale` (d={}) need to have the same number of dimensions!".format(len(origin), len(scale))
    dimensionality = len(origin)

    if fixed_nodes is None:
        fixed_nodes = []

    connected_nodes = _get_unique_nodes(edges)

    if node_positions is None: # assign random starting positions to all nodes
        node_positions_as_array = np.random.rand(len(connected_nodes), dimensionality) * scale + origin
        unique_nodes = connected_nodes

    else:

        if validate_positions:
            # 1) check input dimensionality
            dimensionality_node_positions = np.array(list(node_positions.values())).shape[1]
            assert dimensionality_node_positions == dimensionality, \
                "The dimensionality of values of `node_positions` (d={}) must match the dimensionality of `origin`/ `scale` (d={})!".format(dimensionality_node_positions, dimensionality)

            is_valid = _is_within_bbox(list(node_positions.values()), origin=origin, scale=scale)
            if not np.all(is_valid):
                error_message = "Some given node positions are not within the data range specified by `origin` and `scale`!"
                error_message += "\n\tOrigin : {}, {}".format(*origin)
                error_message += "\n\tScale  : {}, {}".format(*scale)
                error_message += "\nThe following nodes do not fall within this range:"
                for ii, (node, position) in enumerate(node_positions.items()):
                    if not is_valid[ii]:
                        error_message += "\n\t{} : {}".format(node, position)
                error_message += "\nThis error can occur if the graph contains multiple components but some or all node positions are initialised explicitly (i.e. node_positions != None)."
                raise ValueError(error_message)

        # 2) handle discrepancies in nodes listed in node_positions and nodes extracted from edges
        if set(node_positions.keys()) == set(connected_nodes):
            # all starting positions are given;
            # no superfluous nodes in node_positions;
            # nothing left to do
            unique_nodes = connected_nodes
        else:
            # some node positions are provided, but not all
            for node in connected_nodes:
                if not (node in node_positions):
                    warnings.warn("Position of node {} not provided. Initializing to random position within frame.".format(node))
                    node_positions[node] = np.random.rand(2) * scale + origin

            unconnected_nodes = []
            for node in node_positions:
                if not (node in connected_nodes):
                    unconnected_nodes.append(node)
                    fixed_nodes.append(node)
                    # warnings.warn("Node {} appears to be unconnected. The current node position will be kept.".format(node))

            unique_nodes = connected_nodes + unconnected_nodes

        node_positions_as_array = np.array([node_positions[node] for node in unique_nodes])

    total_nodes = len(unique_nodes)

    if isinstance(node_size, (int, float)):
        node_size = node_size * np.ones((total_nodes))
    elif isinstance(node_size, dict):
        node_size = np.array([node_size[node] if node in node_size else 0. for node in unique_nodes])

    adjacency = _edge_list_to_adjacency_matrix(
        edges, edge_weights=edge_weights, unique_nodes=unique_nodes)

    # Forces in FR are symmetric.
    # Hence we need to ensure that the adjacency matrix is also symmetric.
    adjacency = adjacency + adjacency.transpose()

    if fixed_nodes:
        is_mobile = np.array([False if node in fixed_nodes else True for node in unique_nodes], dtype=bool)

        mobile_positions = node_positions_as_array[is_mobile]
        fixed_positions = node_positions_as_array[~is_mobile]

        mobile_node_sizes = node_size[is_mobile]
        fixed_node_sizes = node_size[~is_mobile]

        # reorder adjacency
        total_mobile = np.sum(is_mobile)
        reordered = np.zeros((adjacency.shape[0], total_mobile))
        reordered[:total_mobile, :total_mobile] = adjacency[is_mobile][:, is_mobile]
        reordered[total_mobile:, :total_mobile] = adjacency[~is_mobile][:, is_mobile]
        adjacency = reordered
    else:
        is_mobile = np.ones((total_nodes), dtype=bool)

        mobile_positions = node_positions_as_array
        fixed_positions = np.zeros((0, 2))

        mobile_node_sizes = node_size
        fixed_node_sizes = np.array([])

    if k is None:
        area = np.prod(scale)
        k = np.sqrt(area / float(total_nodes))

    temperatures = _get_temperature_decay(initial_temperature, total_iterations)

    # --------------------------------------------------------------------------------
    # main loop

    for ii, temperature in enumerate(temperatures):
        candidate_positions = _fruchterman_reingold(mobile_positions, fixed_positions,
                                                    mobile_node_sizes, fixed_node_sizes,
                                                    adjacency, temperature, k,
                                                    get_repulsion, get_attraction)
        is_valid = _is_within_bbox(candidate_positions, origin=origin, scale=scale)
        mobile_positions[is_valid] = candidate_positions[is_valid]

    # --------------------------------------------------------------------------------
    # format output

    node_positions_as_array[is_mobile] = mobile_positions

    if np.all(is_mobile):
        node_positions_as_array = _fit_to_frame(node_positions_as_array, origin, scale, pad_by)

    node_positions = dict(zip(unique_nodes, node_positions_as_array))

    return node_positions


def _is_within_bbox(points, origin, scale):
    """Check if each of the given points is within the bounding box given by origin and scale."""
    minima = np.array(origin)
    maxima = minima + np.array(scale)
    return np.all(np.logical_and(points >= minima, points <= maxima), axis=1)


def _get_temperature_decay(initial_temperature, total_iterations, mode='quadratic', eps=1e-9):
    """Compute all temperature values for a given initial temperature and decay model."""
    x = np.linspace(0., 1., total_iterations)
    if mode == 'quadratic':
        y = (x - 1.)**2 + eps
    elif mode == 'linear':
        y = (1. - x) + eps
    else:
        raise ValueError("Argument `mode` one of: 'linear', 'quadratic'.")
    return initial_temperature * y


def _fruchterman_reingold(mobile_positions, fixed_positions,
                          mobile_node_radii, fixed_node_radii,
                          adjacency, temperature, k,
                          get_repulsion, get_attraction):
    """Inner loop of Fruchterman-Reingold layout algorithm."""

    combined_positions = np.concatenate([mobile_positions, fixed_positions], axis=0)
    combined_node_radii = np.concatenate([mobile_node_radii, fixed_node_radii])

    delta = mobile_positions[np.newaxis, :, :] - combined_positions[:, np.newaxis, :]
    distance = np.linalg.norm(delta, axis=-1)

    # alternatively: (hack adapted from igraph)
    if np.sum(distance==0) - np.trace(distance==0) > 0: # i.e. if off-diagonal entries in distance are zero
        warnings.warn("Some nodes have the same position; repulsion between the nodes is undefined.")
        rand_delta = np.random.rand(*delta.shape) * 1e-9
        is_zero = distance <= 0
        delta[is_zero] = rand_delta[is_zero]
        distance = np.linalg.norm(delta, axis=-1)

    # subtract node radii from distances to prevent nodes from overlapping
    distance -= mobile_node_radii[np.newaxis, :] + combined_node_radii[:, np.newaxis]

    # prevent distances from becoming less than zero due to overlap of nodes
    distance[distance <= 0.] = 1e-6 # 1e-13 is numerical accuracy, and we will be taking the square shortly

    with np.errstate(divide='ignore', invalid='ignore'):
        direction = delta / distance[..., None] # i.e. the unit vector

    # calculate forces
    repulsion    = get_repulsion(distance, direction, k)
    attraction   = get_attraction(distance, direction, adjacency, k)
    displacement = attraction + repulsion

    # limit maximum displacement using temperature
    displacement_length = np.linalg.norm(displacement, axis=-1)
    displacement = displacement / displacement_length[:, None] * np.clip(displacement_length, None, temperature)[:, None]

    mobile_positions = mobile_positions + displacement

    return mobile_positions


def _rescale_to_frame(node_positions, origin, scale):
    """Rescale node positions such that all nodes are within the bounding box."""
    node_positions = node_positions.copy() # force copy, as otherwise the `fixed_nodes` argument is effectively ignored
    node_positions -= np.min(node_positions, axis=0)
    # normalize only when nodes are not already aligned, otherwise we divide by zero
    np.divide(node_positions, np.ptp(node_positions, axis=0), where=np.ptp(node_positions, axis=0) != 0, out=node_positions)
    # if nodes are aligned in any one dimension, place in the middle along that dimension
    np.add(node_positions, np.full(node_positions.ndim, 0.5), where=np.ptp(node_positions, axis=0) == 0, out=node_positions)
    node_positions *= scale[None, ...]
    node_positions += origin[None, ...]
    return node_positions


@_rectangle_pack_multiple_components
def get_random_layout(edges, origin=(0,0), scale=(1,1)):
    nodes = _get_unique_nodes(edges)
    return {node : np.random.rand(2) * scale + origin for node in nodes}


@_rectangle_pack_multiple_components
def get_sugiyama_layout(edges, origin=(0, 0), scale=(1, 1), pad_by=0.05, node_size=3, total_iterations=3):
    """'Dot' or Sugiyama node layout.

    Uses the Sugiyama algorithm [Sugiyama1981]_ to compute node positions.
    This function is a wrapper around the SugiyamaLayout class in grandalf.

    Parameters
    ----------
    edges : list
        The edges of the graph, with each edge being represented by a (source node ID, target node ID) tuple.
    origin : tuple, default (0, 0)
        The (float x, float y) coordinates corresponding to the lower left corner of the bounding box specifying the extent of the canvas.
    scale : tuple, default (1, 1)
        The (float x, float y) dimensions representing the width and height of the bounding box specifying the extent of the canvas.
    pad_by : float, default 0.05
        Padding around node positions to reduce clipping of the node artists with the frame,
        and to create space for routing curved edges including self-loops around nodes.
        This results in the following bounding box:

        .. code-block::

           xmin = origin[0] + pad_by * scale[0]
           ymin = origin[1] + pad_by * scale[1]
           xmax = origin[0] + scale[0] - pad_by * scale[0]
           ymax = origin[1] + scale[1] - pad_by * scale[1]

    total_iterations : int, default 50
        Number of iterations.

    Returns
    -------
    node_positions : dict
        Dictionary mapping each node ID to (float x, float y) tuple, the node position.

    References
    ----------
    .. [Sugiyama1981] Sugiyama, K; Tagawa, S; Toda, M (1981) 'Methods for visual understanding of hierarchical system structures',
           IEEE Transactions on Systems, Man, and Cybernetics

    """

    # TODO potentially test that graph is a DAG
    nodes = _get_unique_nodes(edges)
    graph = _get_grandalf_graph(edges, nodes, node_size)

    layout = SugiyamaLayout(graph.C[0])
    layout.init_all()
    layout.draw(total_iterations)

    # extract node positions
    node_positions = dict()
    for layer in layout.layers:
        for vertex in layer:
            if not isinstance(vertex, DummyVertex):
                # The DummyVertex class is used by the sugiyama layout to represent
                # *long* edges, i.e. edges that span over several ranks.
                # For these edges, a DummyVertex is inserted in every inner layer.
                # Here we ignore them, as they are not part of the final layout.
                node_positions[vertex.data] = vertex.view.xy

    # rescale to canvas
    # TODO: by rescaling, we effectively ignore the node_size argument
    nodes, positions = zip(*node_positions.items())
    positions = _rescale_to_frame(np.array(positions), np.array(origin) + pad_by * np.array(scale), (1 - pad_by) * np.array(scale))

    # place roots on top, leaves on bottom
    positions[:, 1] -= origin[1] + scale[1]
    positions[:, 1] *= -1

    return dict(zip(nodes, positions))


def _get_grandalf_graph(edges, nodes, node_size):
    """Construct a grandalf.Graph object from the given edge list, node list, and node sizes."""
    node_to_grandalf_vertex = dict()
    for node in nodes:
        # initialize vertex object
        vertex = Vertex(node)

        # initialize view
        if isinstance(node_size, (int, float)):
            vertex.view = vertex_view(2 * node_size, 2 * node_size)
        elif isinstance(node_size, dict):
            vertex.view = vertex_view(2 * node_size[node], 2 * node_size[node])
        else:
            raise TypeError

        node_to_grandalf_vertex[node] = vertex

    V = list(node_to_grandalf_vertex.values())
    E = [Edge(node_to_grandalf_vertex[source], node_to_grandalf_vertex[target]) for source, target in edges]
    G = Graph(V, E)
    return G


class vertex_view(object):
    def __init__(self, w, h):
        self.w = w
        self.h = h


@_rectangle_pack_multiple_components
def get_radial_tree_layout(edges, origin=(0, 0), scale=(1, 1), pad_by=0.05, node_size=3, total_iterations=3):
    """Radial tree layout.

    Uses the Sugiyama algorithm [Sugiyama1981]_ to compute node positions.
    This function is a wrapper around the SugiyamaLayout class in grandalf.

    Parameters
    ----------
    edges : list
        The edges of the graph, with each edge being represented by a (source node ID, target node ID) tuple.
    origin : tuple, default (0, 0)
        The (float x, float y) coordinates corresponding to the lower left corner of the bounding box specifying the extent of the canvas.
    scale : tuple, default (1, 1)
        The (float x, float y) dimensions representing the width and height of the bounding box specifying the extent of the canvas.
    pad_by : float, default 0.05
        Padding around node positions to reduce clipping of the node artists with the frame,
        and to create space for routing curved edges including self-loops around nodes.
        This results in the following bounding box:

        .. code-block::

           xmin = origin[0] + pad_by * scale[0]
           ymin = origin[1] + pad_by * scale[1]
           xmax = origin[0] + scale[0] - pad_by * scale[0]
           ymax = origin[1] + scale[1] - pad_by * scale[1]

    total_iterations : int, default 50
        Number of iterations.

    Returns
    -------
    node_positions : dict
        Dictionary mapping each node ID to (float x, float y) tuple, the node position.

    References
    ----------
    .. [Sugiyama1981] Sugiyama, K; Tagawa, S; Toda, M (1981) 'Methods for visual understanding of hierarchical system structures',
           IEEE Transactions on Systems, Man, and Cybernetics

    """

    sugiyama_positions = get_sugiyama_layout(edges,
                                             origin           = (0, 0),
                                             scale            = (1, 1),
                                             pad_by           = 0,
                                             node_size        = node_size,
                                             total_iterations = total_iterations
    )

    # determine the size of the largest layer
    nodes_per_layer = dict()
    for node, (_, y) in sugiyama_positions.items():
        if y in nodes_per_layer:
            nodes_per_layer[y] += 1
        else:
            nodes_per_layer[y] = 1
    max_nodes_per_layer = np.max(list(nodes_per_layer.values()))

    # alternatively:
    # y_coordinates = np.array(list(sugiyama_positions.values()))[:, 1]
    # (_, max_nodes_per_layer), = Counter(y_coordinates).most_common(1)

    max_angle = 2 * np.pi * max_nodes_per_layer / (max_nodes_per_layer + 1)
    max_radius = np.min(scale) / 2
    max_radius *= (1 - pad_by) # shrink to make room for node artists, labels, and annotations
    offset = np.array([origin[0] + 0.5 * scale[0], origin[1] + 0.5 * scale[1]])

    node_positions = dict()
    for node, (x, y) in sugiyama_positions.items():
        angle = max_angle * x
        radius = max_radius * (1 - y)
        node_positions[node] = _convert_polar_to_cartesian_coordinates(radius, angle) + offset

    return node_positions


@_rectangle_pack_multiple_components
def get_circular_layout(edges, origin=(0, 0), scale=(1, 1), pad_by=0.05, node_order=None, reduce_edge_crossings=True):
    """Circular node layout.

    By default, this implementation uses a heuristic to arrange the nodes such that the edge crossings are minimised.

    Parameters
    ----------
    edges : list
        The edges of the graph, with each edge being represented by a (source node ID, target node ID) tuple.
    origin : tuple, default (0, 0)
        The (float x, float y) coordinates corresponding to the lower left corner of the bounding box specifying the extent of the canvas.
    scale : tuple, default (1, 1)
        The (float x, float y) dimensions representing the width and height of the bounding box specifying the extent of the canvas.
    pad_by : float, default 0.05
        Padding around node positions to reduce clipping of the node artists with the frame,
        and to create space for routing curved edges including self-loops around nodes.
        This results in the following bounding box:

        .. code-block::

           xmin = origin[0] + pad_by * scale[0]
           ymin = origin[1] + pad_by * scale[1]
           xmax = origin[0] + scale[0] - pad_by * scale[0]
           ymax = origin[1] + scale[1] - pad_by * scale[1]

    node_order : list or None, default None
        The ordering of nodes (left-to-right).
        Implies :code:`reduce_edge_crossings` is :code:`False`.
    reduce_edge_crossings : bool, default True
        If True, attempts to minimize edge crossings via the algorithm outlined in [Baur2005]_.

    Returns
    -------
    node_positions : dict
        Dictionary mapping each node ID to (float x, float y) tuple, the node position.

    See also
    --------
    :py:func:`_get_preorderd_circular_layout`

    References
    ----------
    .. [Baur2005] Baur & Brandes (2005) Crossing reduction in circular layouts.

    """
    nodes = _get_unique_nodes(edges)

    if node_order:
        nodes = [node for node in node_order if node in nodes]
    elif reduce_edge_crossings:
        if not _is_complete(edges):
            # remove self-loops as these should have no bearing on the solution
            edges = [(source, target) for source, target in edges if source != target]
            nodes = _reduce_crossings(edges)
    else:
        try: # emulate networkx behavior
            nodes = sorted(nodes)
        except TypeError: # probably due to mixed node types
            pass

    return _get_preordered_circular_layout(nodes, origin, scale, pad_by)


def _get_preordered_circular_layout(node_order, origin, scale, pad_by=0.05):
    center = np.array(origin) + 0.5 * np.array(scale)
    radius = np.min(scale) / 2
    radius *= (1 - pad_by) # fudge factor to make space for self-loops, annotations, etc
    positions = _get_n_points_on_a_circle(center, radius, len(node_order), start_angle=0)
    return dict(zip(node_order, positions))


def _is_complete(edges):
    """Check if the graph is fully connected."""
    nodes = _get_unique_nodes(edges)
    minimal_complete_graph = list(combinations(nodes, 2))

    if len(edges) < len(minimal_complete_graph):
        return False

    for edge in minimal_complete_graph:
        if (edge not in edges) and (edge[::-1] not in edges):
            return False

    return True


def _reduce_crossings(edges):
    """Implements Baur & Brandes (2005) Crossing reduction in circular layouts."""
    adjacency_list = _edge_list_to_adjacency_list(edges, directed=False)
    node_order = _initialize_node_order(adjacency_list)
    node_order = _optimize_node_order(adjacency_list, node_order)
    return node_order


def _initialize_node_order(node_to_neighbours):
    """Implements "Connectivity & Crossings" variant from Baur & Brandes (2005)."""
    nodes = list(node_to_neighbours.keys())
    ordered = nodes[:1]
    remaining = nodes[1:]
    closed_edges = []
    start, = ordered
    open_edges = set([(start, neighbour) for neighbour in node_to_neighbours[start]])

    while remaining:
        minimum_unplaced_neighbours = np.inf
        maximum_placed_neighbours = 0
        for ii, node in enumerate(remaining):
            placed_neighbours = len([neighbour for neighbour in node_to_neighbours[node] if neighbour in ordered])
            unplaced_neighbours = len([neighbour for neighbour in node_to_neighbours[node] if neighbour not in ordered])
            if (placed_neighbours > maximum_placed_neighbours) or \
               ((placed_neighbours == maximum_placed_neighbours) and (unplaced_neighbours < minimum_unplaced_neighbours)):
                maximum_placed_neighbours = placed_neighbours
                minimum_unplaced_neighbours = unplaced_neighbours
                selected = node
                selected_idx = ii

        remaining.pop(selected_idx)
        closed_edges = set([(node, selected) for node in node_to_neighbours[selected] if node in ordered])
        open_edges -= closed_edges
        a = _get_total_crossings(ordered + [selected] + remaining, closed_edges, open_edges)
        b = _get_total_crossings([selected] + ordered + remaining, closed_edges, open_edges)
        if a < b:
            ordered.append(selected)
        else:
            ordered.insert(0, selected)
        open_edges |= set([(selected, node) for node in node_to_neighbours[selected] if node not in ordered])

    return ordered


def _get_total_crossings(node_order, edges1, edges2=None):
    """Compute the number of crossings for a given node order."""
    if edges2 is None:
        edges2 = edges1
    total_crossings = 0
    for edge1, edge2 in itertools.product(edges1, edges2):
        total_crossings += int(_is_cross(node_order, edge1, edge2))
    return total_crossings


def _is_cross(node_order, edge_1, edge_2):
    """Check if two edges cross, given the node order."""
    (s1, t1), = np.where(np.in1d(node_order, edge_1, assume_unique=True))
    (s2, t2), = np.where(np.in1d(node_order, edge_2, assume_unique=True))

    if (s1 < s2 < t1 < t2) or (s2 < s1 < t2 < t2):
        return True
    else:
        return False


def _optimize_node_order(node_to_neighbours, node_order=None, max_iterations=100):
    """Implement circular sifting as outlined in Baur & Brandes (2005)."""
    if node_order is None:
        node_order = list(node_to_neighbours.keys())
    node_order = np.array(node_order)

    node_to_edges = dict()
    for node, neighbours in node_to_neighbours.items():
        node_to_edges[node] = [(node, neighbour) for neighbour in neighbours]

    undirected_edges = set()
    for edges in node_to_edges.values():
        for edge in edges:
            if edge[::-1] not in undirected_edges:
                undirected_edges.add(edge)
    total_crossings = _get_total_crossings(node_order, undirected_edges)

    total_nodes = len(node_order)
    for iteration in range(max_iterations):
        previous_best = total_crossings
        for u in node_order.copy():
            best_order = node_order.copy()
            minimum_crossings = total_crossings
            (ii,), = np.where(node_order == u)
            for jj in range(ii+1, ii+total_nodes):
                v = node_order[jj%total_nodes]
                cuv = _get_total_crossings(node_order, node_to_edges[u], node_to_edges[v])
                node_order = _swap_values(node_order, u, v)
                cvu = _get_total_crossings(node_order, node_to_edges[u], node_to_edges[v])
                total_crossings = total_crossings - cuv + cvu
                if total_crossings < minimum_crossings:
                    best_order = node_order.copy()
                    minimum_crossings = total_crossings
            node_order = best_order
            total_crossings = minimum_crossings

        improvement = previous_best - total_crossings
        if not improvement:
            break

    if (iteration + 1) == max_iterations:
        warnings.warn("Maximum number of iterations reached. Aborting further node layout optimisations.")

    return node_order


def _swap_values(arr, value_1, value_2):
    """Swap all occurrences of two values in a given array."""
    arr = arr.copy()
    is_value_1 = arr == value_1
    is_value_2 = arr == value_2
    arr[is_value_1] = value_2
    arr[is_value_2] = value_1
    return arr


def _reduce_node_overlap(node_positions, origin, scale, fixed_nodes=None, eta=0.1, total_iterations=10):
    """Use a constrained version of Lloyd's algorithm to move nodes apart from each other.

    References
    ----------
    https://en.wikipedia.org/wiki/Lloyd%27s_algorithm
    """

    unique_nodes = list(node_positions.keys())
    positions = np.array(list(node_positions.values()))

    if fixed_nodes:
        is_mobile = np.array([False if node in fixed_nodes else True for node in unique_nodes], dtype=bool)
    else:
        is_mobile = np.ones((len(unique_nodes)), dtype=bool)

    for _ in range(total_iterations):
        centroids = _get_voronoi_centroids(positions)
        delta = centroids - positions
        new = positions + eta * delta
        # constrain Lloyd's algorithm by only updating positions where the new position is within the bbox
        valid = _is_within_bbox(new, origin, scale)
        mask = np.logical_and(valid, is_mobile)
        positions[mask] = new[mask]

    return dict(zip(unique_nodes, positions))


def _remove_node_overlap(node_positions, node_size, origin, scale, fixed_nodes=None, tolerance=1e-6, maximum_iterations=100):
    """Uses a constrained variation of Lloyd's algorithm to move nodes apart from each other until none overlap."

    References
    ----------
    https://en.wikipedia.org/wiki/Lloyd%27s_algorithm

    """

    unique_nodes = list(node_positions.keys())
    positions = np.array(list(node_positions.values()))
    radii = np.array([node_size[node] for node in node_positions])

    if fixed_nodes:
        is_mobile = np.array([False if node in fixed_nodes else True for node in unique_nodes], dtype=bool)
    else:
        is_mobile = np.ones((len(unique_nodes)), dtype=bool)

    minimum_distances = radii[np.newaxis, :] + radii[:, np.newaxis]
    minimum_distances[np.diag_indices_from(minimum_distances)] = 0 # ignore distances to self

    # Initialize the first loop.
    distances = cdist(positions, positions)
    displacements = np.max(np.clip(minimum_distances - distances, 0, None), axis=-1)

    ctr = 0
    while np.any(displacements > tolerance) & (ctr < maximum_iterations):
        centroids = _get_voronoi_centroids(positions)

        # Compute the direction from each point towards its corresponding Voronoi centroid.
        deltas = centroids - positions
        magnitudes = np.linalg.norm(deltas, axis=-1)
        directions = deltas / magnitudes[:, np.newaxis]

        # Mask NaNs that arise if the magnitude is zero, i.e. the point is already center of the Voronoi cell.
        directions[np.isnan(directions)] = 0

        # Step into the direction of the centroid.
        # Clipping prevents overshooting of the centroid when stepping into the direction of the centroid.
        # We step by half the displacement as the other overlapping point will be moved in approximately the opposite direction.
        new = positions + np.clip(0.5 * displacements, None, magnitudes)[:, np.newaxis] * directions

        # Constrain Lloyd's algorithm by only updating positions where the new position is within the bbox.
        valid = _is_within_bbox(new, origin, scale)
        mask = np.logical_and(valid, is_mobile)
        positions[mask] = new[mask]

        # Initialize next loop.
        distances = cdist(positions, positions)
        displacements = np.max(np.clip(minimum_distances - distances, 0, None), axis=-1)
        ctr += 1

    return dict(zip(unique_nodes, positions))


def _get_voronoi_centroids(positions):
    """Construct a Voronoi diagram from the given node positions and determine the center of each cell."""
    voronoi = Voronoi(positions)
    centroids = np.zeros_like(positions)
    for ii, idx in enumerate(voronoi.point_region):
        region = [jj for jj in voronoi.regions[idx] if jj != -1] # i.e. ignore points at infinity; TODO: compute correctly clipped regions
        centroids[ii] = _get_centroid(voronoi.vertices[region])
    return centroids


def _clip_to_frame(positions, origin, scale):
    """Prevent node positions from leaving the bounding box given by origin and scale."""
    origin = np.array(origin)
    scale = np.array(scale)
    for ii, (minimum, maximum) in enumerate(zip(origin, origin+scale)):
        positions[:, ii] = np.clip(positions[:, ii], minimum, maximum)
    return positions


def _get_centroid(polygon):
    """Compute the centroid of a polygon."""
    # TODO: formula may be incorrect; correct one here:
    # https://en.wikipedia.org/wiki/Centroid#Of_a_polygon
    return np.mean(polygon, axis=0)


@_arrange_multiple_components_side_by_side
def get_linear_layout(edges, origin=(0, 0), scale=(1, 1), pad_by=0.05, node_order=None, reduce_edge_crossings=True):
    """Linear node layout.

    If :code:`reduce_edge_crossings` is set to :code:`True`, the algorithm
    attempts to minimize edge crossings via the algorithm outlined in [Baur2005]_.

    Parameters
    ----------
    edges : list
        The edges of the graph, with each edge being represented by a (source node ID, target node ID) tuple.
    origin : tuple, default (0, 0)
        The (float x, float y) coordinates corresponding to the lower left corner of the bounding box specifying the extent of the canvas.
    scale : tuple, default (1, 1)
        The (float x, float y) dimensions representing the width and height of the bounding box specifying the extent of the canvas.
    pad_by : float, default 0.05
        Padding around node positions to reduce clipping of the node artists with the frame,
        and to create space for routing curved edges including self-loops around nodes.
        This results in the following bounding box:

        .. code::

            xmin = origin[0] + pad_by * scale[0]
            ymin = origin[1] + pad_by * scale[1]
            xmax = origin[0] + scale[0] - pad_by * scale[0]
            ymax = origin[1] + scale[1] - pad_by * scale[1]

    node_order : list or None, default None
        The ordering of nodes (left-to-right).
        Implies :code:`reduce_edge_crossings` is :code:`False`.
    reduce_edge_crossings : bool, default True
        If True, attempts to minimize edge crossings via the algorithm outlined in [Baur2005]_.

    Returns
    -------
    node_positions : dict
        Dictionary mapping each node ID to (float x, float y) tuple, the node position.

    See also
    --------
    _get_preordered_linear_layout

    References
    ----------
    .. [Baur2005] Baur & Brandes (2005) Crossing reduction in circular layouts.

    """

    nodes = _get_unique_nodes(edges)

    if node_order:
        nodes = [node for node in node_order if node in nodes]
    elif reduce_edge_crossings:
        if not _is_complete(edges):
            # remove self-loops as these should have no bearing on the solution
            edges = [(source, target) for source, target in edges if source != target]
            nodes = _reduce_crossings(edges)
            # The algorithm in _reduce_crossings assumes that nodes are arranged in a circle.
            # As a consequence, when nodes are arranged along a line, a community may be split
            # with members on both ends of the line.
            # Here we find a better split by minimising the total edge length.
            nodes = _minimize_total_edge_length(nodes, edges)
    else:
        try:
            nodes = sorted(nodes)
        except TypeError: # probably due to mixed node types
            pass

    return _get_preordered_linear_layout(nodes, origin, scale, pad_by)


def _get_preordered_linear_layout(node_order, origin, scale, pad_by=0.05):
    total_nodes = len(node_order)
    x = np.linspace(origin[0] + pad_by * scale[0], origin[0] + (1 - pad_by) * scale[0], total_nodes)
    y = np.full(total_nodes, origin[1] + 0.5 * scale[1])
    return dict(zip(node_order, np.c_[x, y]))


def _minimize_total_edge_length(nodes, edges):
    total_iterations = len(nodes)
    output = nodes
    optimum = np.inf
    for ii in range(total_iterations):
        node_to_position = dict(zip(nodes, range(len(nodes))))
        total_edge_length = np.sum([np.abs(node_to_position[source] - node_to_position[target]) for (source, target) in edges])
        if total_edge_length < optimum:
            output = nodes
            optimum = total_edge_length
        nodes = np.r_[nodes[1:], nodes[0]]
    return output


def get_bipartite_layout(edges, nodes=None, subsets=None, origin=(0, 0), scale=(1, 1), pad_by=0.05, reduce_edge_crossings=True):
    """Bipartite node layout.

    By default, this implementation uses a heuristic to arrange the nodes such that the edge crossings are reduced.

    Parameters
    ----------
    edges : list
        The edges of the graph, with each edge being represented by a (source node ID, target node ID) tuple.
    subsets : list
        The two layers of the graph. If None, a two-coloring is used to separate the nodes into two subsets.
        However, if the graph consists of multiple components, this partitioning into two layers is ambiguous, as multiple solutions exist.
    origin : tuple
        The (float x, float y) coordinates corresponding to the lower left corner of the bounding box specifying the extent of the canvas.
    scale : tuple
        The (float x, float y) dimensions representing the width and height of the bounding box specifying the extent of the canvas.
    pad_by : float, default 0.05
        Padding around node positions to reduce clipping of the node artists with the frame,
        and to create space for routing curved edges including self-loops around nodes.
        This results in the following bounding box:

        .. code::

            xmin = origin[0] + pad_by * scale[0]
            ymin = origin[1] + pad_by * scale[1]
            xmax = origin[0] + scale[0] - pad_by * scale[0]
            ymax = origin[1] + scale[1] - pad_by * scale[1]

    reduce_edge_crossings : bool, default True
        If True, attempts to reduce edge crossings via the algorithm outlined in [Eades1994]_.

    Returns
    -------
    node_positions : dict
        Dictionary mapping each node ID to (float x, float y) tuple, the node position.

    References
    ----------
    .. [Eades1994] Eades & Wormald (1994) Edge crossings in drawings of bipartite graphs.

    """

    adjacency_list = _edge_list_to_adjacency_list(edges, directed=False)

    if nodes:
        for node in nodes:
            adjacency_list.setdefault(node, set())

    if subsets:
        left, right = subsets
    else:
        if len(_get_connected_components(adjacency_list)) > 1:
            import warnings
            msg = "The graph consistst of multiple components, and hence the partitioning into two subsets/layers is ambiguous!"
            msg += "\n"
            msg += "Use the `subsets` argument to explicitly specify the desired partitioning."
            warnings.warn(msg)
        left, right = _get_bipartite_sets(adjacency_list)

    if reduce_edge_crossings:
        if not _is_complete_bipartite(edges, left, right):
            left, right = _reduce_crossings_bipartite(adjacency_list, left, right)

    # shrink frame to apply padding
    origin = np.array(origin) + pad_by * np.array(scale)
    scale = np.array(scale) * (1 - 2 * pad_by)

    # determine the spacing between nodes within a subset
    if len(edges) == 1:
        spacing = 1.
    elif len(left) > len(right):
        spacing = scale[1] / (len(left) - 1)
    else:
        spacing = scale[1] / (len(right) - 1)

    # set node positions
    node_positions = dict()
    for subset, xx in ((left, origin[0]), (right, origin[0] + scale[0])):
        y = spacing * np.arange(len(subset))
        y -= np.mean(y)
        y += scale[1] / 2
        for node, yy in zip(subset, y):
            node_positions[node] = (xx, yy)

    return node_positions


def _get_bipartite_sets(adjacency_list):
    colour = _get_two_colouring(adjacency_list)
    left = [node for node in colour if colour[node] == 0]
    right = [node for node in colour if colour[node] == 1]
    return left, right


def _get_two_colouring(adjacency_list):
    # Adapted from networkx.algorithms.bipartite.color.
    colour = dict()
    for node in adjacency_list:
        if (node in colour):
            continue
        elif len(adjacency_list[node]) == 0:
            colour[node] = 0
        else:
            queue = [node]
            colour[node] = 1
            while queue:
                node = queue.pop()
                for neighbour in adjacency_list[node]:
                    if neighbour in colour:
                        if colour[neighbour] == colour[node]:
                            raise Exception("Graph is not bipartite.")
                    else:
                        colour[neighbour] = 1 - colour[node]
                        queue.append(neighbour)
    return colour


def _is_complete_bipartite(edges, left, right):
    """Check if the bipartite graph is fully connected."""
    minimal_complete_graph = list(product(left, right))

    if len(edges) < len(minimal_complete_graph):
        return False

    for edge in minimal_complete_graph:
        if (edge not in edges) and (edge[::-1] not in edges):
            return False

    return True


def _reduce_crossings_bipartite(adjacency_list, left, right):
    """Reduce the number of crossings in a bipartite graph using the median heuristic proposed in Eades & Wormald (1994)."""

    left_ranks = {node : ii for ii, node in enumerate(left)}
    right_ranks = dict()
    for node in right:
        neighbours = adjacency_list[node]
        if neighbours:
            right_ranks[node] = np.median([left_ranks[neighbour] for neighbour in neighbours])
        else:
            right_ranks[node] = 0

    # TODO: break ties.
    # If one node has an even number of neighbours and the other an odd number, than the odd one should have the lower rank.
    # For the other two cases (even/even, odd/odd), no tie-break procedure seems specified in the paper.

    return left, sorted(right_ranks, key=right_ranks.get)


def get_multipartite_layout(edges, layers, layer_positions=None, origin=(0, 0), scale=(1, 1), pad_by=0.05, reduce_edge_crossings=True, uniform_node_spacing=True):
    """Layered node layout for a multipartite graph.

    By default, this implementation uses a heuristic to arrange the nodes such that the edge crossings are reduced.

    Parameters
    ----------
    edges : list
        The edges of the graph, with each edge being represented by a (source node ID, target node ID) tuple.
    layers : list
        List of node subsets, one for each layer of the graph.
    layer_positions : list, default None
        A list of x-coordinates, one for each layer.
        If None provided, layers are placed evenly between origin[0] and origin[0] + scale[0].
    origin : tuple, default (0, 0)
        The (float x, float y) coordinates corresponding to the lower left corner of the bounding box specifying the extent of the canvas.
    scale : tuple, default (1, 1)
        The (float x, float y) dimensions representing the width and height of the bounding box specifying the extent of the canvas.
    pad_by : float, default 0.05
        Padding around node positions to reduce clipping of the node artists with the frame,
        and to create space for routing curved edges including self-loops around nodes.
        This results in the following bounding box:

        .. code::

            xmin = origin[0] + pad_by * scale[0]
            ymin = origin[1] + pad_by * scale[1]
            xmax = origin[0] + scale[0] - pad_by * scale[0]
            ymax = origin[1] + scale[1] - pad_by * scale[1]

    reduce_edge_crossings : bool, default True
        If True, attempts to reduce edge crossings via the algorithm outlined in [Eades1994]_.
    uniform_node_spacing : bool, default True
        If True, the spacing between nodes is uniform across layers.
        Otherwise, nodes in each layer are distributed evenly across the full height of the canvas.

    Returns
    -------
    node_positions : dict
        Dictionary mapping each node ID to (float x, float y) tuple, the node position.

    References
    ----------
    .. [Eades1994] Eades & Wormald (1994) Edge crossings in drawings of bipartite graphs.

    """

    # shrink frame to apply padding
    origin = np.array(origin) + pad_by * np.array(scale)
    scale = np.array(scale) * (1 - 2 * pad_by)

    # set the space between nodes
    if uniform_node_spacing:
        try:
            spacing = scale[1] / (np.max([len(layer) for layer in layers]) - 1)
            node_spacings =  [spacing * np.ones_like(layer, dtype=float) for layer in layers]
        except ZeroDivisionError:
            # The graph has at most a single edge between each pair of layers.
            spacing = 1
            node_spacings = [spacing * np.ones_like(layer, dtype=float) for layer in layers]
    else:
        node_spacings = []
        for ii, layer in enumerate(layers):
            if len(layer) > 1:
                node_spacings.append(1./(len(layer) - 1) * np.ones_like(layer, float))
            else:
                node_spacings.append(np.ones_like(layer, float))

    # set the space between layers
    if layer_positions is None:
        layer_positions = np.linspace(origin[0], origin[0] + scale[0], len(layers))
    else:
        if (np.min(layer_positions) < origin[0]) or (np.max(layer_positions) > origin[0] + scale[0]):
            import warnings
            warnings.warn("Some layer positions are outside the bounding box defined by `origin` and `scale`.")

    # fix positions of nodes in first layer
    node_positions = dict()
    node_positions.update(_get_node_positions_within_layer(layers[0], node_spacings[0], layer_positions[0], origin, scale))

    # assign the position of nodes in subsequent layers
    total_layers = len(layers)
    for ii in range(1, total_layers):
        left = layers[ii-1]
        right = layers[ii]
        union = set(left) | set(right)
        edges_between_layers = [(source, target) for (source, target) in edges if (source in union) and (target in union)]
        adjacency_list = _edge_list_to_adjacency_list(edges_between_layers, directed=False)

        # add unconnected nodes
        for node in left:
            adjacency_list.setdefault(node, set())
        for node in right:
            adjacency_list.setdefault(node, set())

        if reduce_edge_crossings:
            if not _is_complete_bipartite(edges, left, right):
                left, right = _reduce_crossings_bipartite(adjacency_list, left, right)

        node_positions.update(_get_node_positions_within_layer(right, node_spacings[ii], layer_positions[ii], origin, scale))

    return node_positions


def _get_node_positions_within_layer(node_order, node_spacing, layer_position, origin, scale):
    node_positions = dict()
    xx = layer_position
    y = node_spacing * np.arange(len(node_spacing))
    y -= np.mean(y)
    y += origin[1] + scale[1] / 2
    for node, yy in zip(node_order, y):
        node_positions[node] = np.array((xx, yy))
    return node_positions


def get_shell_layout(edges, shells, radii=None, origin=(0, 0), scale=(1, 1), pad_by=0.05, reduce_edge_crossings=True):
    """Shell layout.

    This is a wrapper around `get_multipartite_layout` that arranges nodes in shells around a center instead of in layers.

    Parameters
    ----------
    edges : list
        The edges of the graph, with each edge being represented by a (source node ID, target node ID) tuple.
    shells : list
        List of node subsets, one for each shell of the graph.
    radii : list, default None
        List of radii, one for each shell.
        If None, radii are chosen such that the shells are evenly spaced within the bounding box defined by origin and scale.
    origin : tuple
        The (float x, float y) coordinates corresponding to the lower left corner of the bounding box specifying the extent of the canvas.
    scale : tuple
        The (float x, float y) dimensions representing the width and height of the bounding box specifying the extent of the canvas.
    pad_by : float, default 0.05
        Padding around node positions to reduce clipping of the node artists with the frame,
        and to create space for routing curved edges including self-loops around nodes.
        This results in the following bounding box:

        .. code::

            xmin = origin[0] + pad_by * scale[0]
            ymin = origin[1] + pad_by * scale[1]
            xmax = origin[0] + scale[0] - pad_by * scale[0]
            ymax = origin[1] + scale[1] - pad_by * scale[1]

    reduce_edge_crossings : bool, default True
        If True, attempts to reduce edge crossings via the algorithm outlined in [Eades1994]_.

    Returns
    -------
    node_positions : dict
        Dictionary mapping each node ID to (float x, float y) tuple, the node position.

    References
    ----------
    .. [Eades1994] Eades & Wormald (1994) Edge crossings in drawings of bipartite graphs.

    """

    if radii is None:
        if len(shells[0]) == 1:
            # Innermost shell consists of a single node, and hence should have no size.
            radii = np.linspace(0, (1 - pad_by) * np.min(scale) / 2, len(shells))
        else:
            # Innermost shell consists of multiple nodes and should hence have a non-zero radius.
            radii = np.linspace(0, (1 - pad_by) * np.min(scale) / 2, len(shells) + 1)[1:]

    relative_radii = np.array(radii) / np.max(radii)
    multipartite_positions = get_multipartite_layout(
        edges, shells, layer_positions=relative_radii,
        reduce_edge_crossings=reduce_edge_crossings,
        uniform_node_spacing=False,
        pad_by=0,
    )

    node_to_shell = {node : shell for shell in shells for node in shell}

    max_radius = np.max(radii)
    center = np.array([origin[0] + 0.5 * scale[0], origin[1] + 0.5 * scale[1]])

    node_positions = dict()
    for node, (x, y) in multipartite_positions.items():
        shell = node_to_shell[node]
        max_angle = 2 * np.pi * len(shell) / (len(shell) + 1)
        angle = max_angle * y
        radius = max_radius * x
        node_positions[node] = _convert_polar_to_cartesian_coordinates(radius, angle) + center

    return node_positions


@_rectangle_pack_multiple_components
def get_community_layout(edges, node_to_community, origin=(0, 0), scale=(1, 1), pad_by=0.05):
    """Community node layout for modular graphs.

    This layout is based on ideas presented in [Traud2009], and involves the following steps:

      1. Position the communities with respect to each other: create a new, weighted graph, where each node corresponds to a community, and the weights correspond to the number of edges between communities, and compute a layout for this community graph.
      2. Position the nodes within each community: for each community, create a subgraph. Find a layout for the subgraph.
      3. Combine positions from step 1 & 2 such that node positions are dominated by the corresponding community position and refined by their position within the community subgraph.
      4. Rotate communities to reduce the length of inter-community edges.

    Parameters
    ----------
    edges : list
        The edges of the graph, with each edge being represented by a (source node ID, target node ID) tuple.
    node_to_community : dict
        The network partition, which maps each node ID to a community ID.
    origin : tuple, default (0, 0)
        The (float x, float y) coordinates corresponding to the lower left corner of the bounding box specifying the extent of the canvas.
    scale : tuple, default (1, 1)
        The (float x, float y) dimensions representing the width and height of the bounding box specifying the extent of the canvas.
    pad_by : float, default 0.05
        Padding around node positions to reduce clipping of the node artists with the frame,
        and to create space for routing curved edges including self-loops around nodes.
        This results in the following bounding box:

        .. code::

            xmin = origin[0] + pad_by * scale[0]
            ymin = origin[1] + pad_by * scale[1]
            xmax = origin[0] + scale[0] - pad_by * scale[0]
            ymax = origin[1] + scale[1] - pad_by * scale[1]

    Returns
    -------
    node_positions : dict
        Dictionary mapping each node ID to (float x, float y) tuple, the node position.

    References
    ----------
    .. [Traud2009] Traud et al. (2009) Visualization of communities in networks. https://doi.org/10.1063/1.3194108

    """

    # assert that there multiple communities in the graph; otherwise abort
    nodes = _get_unique_nodes(edges)
    communities = set([node_to_community[node] for node in nodes])
    if len(communities) < 2:
        warnings.warn("Graph contains a single community. Unable to compute a community layout. Computing spring layout instead.")
        return get_fruchterman_reingold_layout(edges, origin=origin, scale=scale, pad_by=pad_by)

    # assert that node_to_community is non-redundant,
    # i.e. only contains nodes that are also present in edges
    node_to_community = {node : node_to_community[node] for node in nodes}

    community_size = _get_community_sizes(node_to_community, scale)
    community_centroids = _get_community_positions(edges, node_to_community, community_size, origin, scale, pad_by)
    relative_node_positions = _get_within_community_positions(edges, node_to_community)
    node_positions = _combine_positions(node_to_community, community_centroids, community_size, relative_node_positions)
    node_positions = _rotate_communities(edges, node_to_community, community_centroids, node_positions)

    return node_positions


def _get_community_sizes(node_to_community, scale):
    """Compute the area of the canvas reserved for each community."""
    total_nodes = len(node_to_community)
    max_radius = np.linalg.norm(scale) / 2
    scalar = max_radius / total_nodes # this is the worst case scenario, where all comunities are lined up like beads on a string; may warrant revisiting
    community_to_nodes = _invert_dict(node_to_community)
    community_size = {community : len(nodes) * scalar for community, nodes in community_to_nodes.items()}
    return community_size


def _get_community_positions(edges, node_to_community, community_size, origin, scale, pad_by):
    """Compute a centroid position for each community."""
    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    between_community_edges = _find_between_community_edges(edges, node_to_community)

    # find layout for communities
    return get_fruchterman_reingold_layout(
        list(between_community_edges.keys()), edge_weight=between_community_edges,
        node_size=community_size, origin=origin, scale=scale, pad_by=pad_by,
    )


def _find_between_community_edges(edges, node_to_community):
    """Convert the graph into a weighted network of communities."""
    between_community_edges = dict()

    for (ni, nj) in edges:
        ci = node_to_community[ni]
        cj = node_to_community[nj]

        if ci != cj:
            if (ci, cj) in between_community_edges:
                between_community_edges[(ci, cj)] += 1
            elif (cj, ci) in between_community_edges:
                # only compute the undirected graph
                between_community_edges[(cj, ci)] += 1
            else:
                between_community_edges[(ci, cj)] = 1

    return between_community_edges


def _get_within_community_positions(edges, node_to_community):
    """Positions nodes within communities."""
    community_to_nodes = _invert_dict(node_to_community)
    node_positions = dict()
    for community, nodes in community_to_nodes.items():
        if len(nodes) > 1:
            subgraph = _get_subgraph(edges, list(nodes))
            if subgraph:
                subgraph_node_positions = get_fruchterman_reingold_layout(
                    subgraph, nodes=nodes, origin=np.array([-1, -1]), scale=np.array([2, 2]))
                node_positions.update(subgraph_node_positions)
            else:
                warnings.warn(f"There are no connections within community {community}. The placement of of nodes within this community is arbitrary.")
                node_positions.update({node : np.random.rand(2) * 2 + np.array([-1, -1]) for node in nodes})
        elif len(nodes) == 1:
            node_positions.update({nodes.pop() : np.array([0., 0.])})
    return node_positions


def _combine_positions(node_to_community, community_centroids, community_size, relative_node_positions):
    node_positions = dict()
    for node, community in node_to_community.items():
        xy = community_centroids[community]
        delta = relative_node_positions[node] * community_size[community]
        node_positions[node] = xy + delta
    return node_positions


def _rotate_communities(edges, node_to_community, community_centroids, node_positions, step_size=0.1, max_iterations=200):

    between_community_edges = [(source, target) for (source, target) in edges \
                               if node_to_community[source] != node_to_community[target]]

    for _ in range(max_iterations):

        # compute torques
        community_torque = {community : 0 for community in set(list(node_to_community.values()))}
        for (source, target) in between_community_edges:
            # source
            community = node_to_community[source]
            r = community_centroids[community] - node_positions[source]
            delta = node_positions[target] - node_positions[source]
            F = delta * np.linalg.norm(delta) # direction * distance**2
            community_torque[community] += np.cross(r, F)

            # target
            community = node_to_community[target]
            r = community_centroids[community] - node_positions[target]
            delta = node_positions[source] - node_positions[target]
            F = delta * np.linalg.norm(delta)
            community_torque[community] += np.cross(r, F)

        # update node positions
        for node, community in node_to_community.items():
            node_positions[node] = _rotate(step_size * -community_torque[community],
                                           node_positions[node],
                                           community_centroids[community])

    return node_positions


def _rotate(angle, points, origin=(0, 0)):
    # https://stackoverflow.com/a/58781388/2912349
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    origin = np.atleast_2d(origin)
    points = np.atleast_2d(points)
    return np.squeeze((R @ (points.T-origin.T) + origin.T).T)


@_rectangle_pack_multiple_components
def get_geometric_layout(edges, edge_length, node_size=0., tol=1e-3, origin=(0, 0), scale=(1, 1), pad_by=0.05):
    """Node layout for defined edge lengths but unknown node positions.

    Node positions are determined through non-linear optimisation: the
    total distance between nodes is maximised subject to the constraint
    imposed by the edge lengths, which are used as upper bounds.
    If provided, node sizes are used to set lower bounds to minimise collisions.

    ..note:: This implementation is slow.

    Parameters
    ----------
    edges : list
        The edges of the graph, with each edge being represented by a (source node ID, target node ID) tuple.
    edge_lengths : dict
        Mapping of edges to their lengths.
    node_size : scalar or dict, default 0.
        Size (radius) of nodes.
        Providing the correct node size minimises the overlap of nodes in the graph,
        which can otherwise occur if there are many nodes, or if the nodes differ considerably in size.
    tolerance : float, default 1e-3
        The tolerance of the cost function. Small values increase the accuracy, large values improve the computation time.
    origin : tuple, default (0, 0)
        The (float x, float y) coordinates corresponding to the lower left corner of the bounding box specifying the extent of the canvas.
    scale : tuple, default (1, 1)
        The (float x, float y) dimensions representing the width and height of the bounding box specifying the extent of the canvas.
    pad_by : float, default 0.05
        Padding around node positions to reduce clipping of the node artists with the frame,
        and to create space for routing curved edges including self-loops around nodes.
        This results in the following bounding box:

        .. code-block::

           xmin = origin[0] + pad_by * scale[0]
           ymin = origin[1] + pad_by * scale[1]
           xmax = origin[0] + scale[0] - pad_by * scale[0]
           ymax = origin[1] + scale[1] - pad_by * scale[1]

    Returns
    -------
    node_positions : dict
        Dictionary mapping each node ID to (float x, float y) tuple, the node position.

    """

    # TODO: assert triangle inequality is not violated.
    # HOLD: probably not necessary, as minimisation can still proceed when triangle inequality is violated.

    # assert that the edges fit within the canvas dimensions
    width, height = scale
    max_length = np.sqrt(width**2 + height**2)
    too_long = dict()
    for edge, length in edge_length.items():
        if length > max_length:
            too_long[edge] = length
    if too_long:
        msg = f"The following edges exceed the dimensions of the canvas (`scale={scale}`):"
        for edge, length in too_long.items():
            msg += f"\n\t{edge} : {length}"
        msg += "\nEither increase the `scale` parameter, or decrease the edge lengths."
        raise ValueError(msg)

    # ensure that graph is bi-directional
    edges = edges + [(target, source) for (source, target) in edges] # forces copy
    edges = list(set(edges))

    # upper bound: pairwise distance matrix with unknown distances set to the maximum possible distance given the canvas dimensions

    lengths = []
    for (source, target) in edges:
        if (source, target) in edge_length:
            lengths.append(edge_length[(source, target)])
        else:
            lengths.append(edge_length[(target, source)])

    sources, targets = zip(*edges)
    nodes = sources + targets
    unique_nodes = set(nodes)
    indices = range(len(unique_nodes))
    node_to_idx = dict(zip(unique_nodes, indices))
    source_indices = [node_to_idx[source] for source in sources]
    target_indices = [node_to_idx[target] for target in targets]

    total_nodes = len(unique_nodes)
    distance_matrix = np.full((total_nodes, total_nodes), max_length)
    distance_matrix[source_indices, target_indices] = lengths
    distance_matrix[np.diag_indices(total_nodes)] = 0
    upper_bounds = squareform(distance_matrix)

    # lower bound: sum of node sizes
    if isinstance(node_size, (int, float)):
        sizes = node_size * np.ones((total_nodes))
    elif isinstance(node_size, dict):
        sizes = np.array([node_size[node] if node in node_size else 0. for node in unique_nodes])

    sum_of_node_sizes = sizes[np.newaxis, :] + sizes[:, np.newaxis]
    sum_of_node_sizes -= np.diag(np.diag(sum_of_node_sizes)) # squareform requires zeros on diagonal
    lower_bounds = squareform(sum_of_node_sizes)
    invalid = lower_bounds > upper_bounds
    lower_bounds[invalid] = upper_bounds[invalid] - 1e-8

    # For an extended discussion of this cost function and alternatives see:
    # https://stackoverflow.com/q/75137677/2912349
    def cost_function(positions):
        return 1 / np.sum(np.log(pdist(positions.reshape((-1, 2))) + 1))

    def constraint_function(positions):
        positions = np.reshape(positions, (-1, 2))
        return pdist(positions)

    initial_positions = _initialise_geometric_node_layout(edges, edge_length)
    nonlinear_constraint = NonlinearConstraint(constraint_function, lb=lower_bounds, ub=upper_bounds, jac='2-point')
    result = minimize(
        cost_function,
        initial_positions.flatten(),
        method='SLSQP',
        jac='2-point',
        constraints=[nonlinear_constraint],
        options=dict(ftol=tol),
    )

    if not result.success:
        print("Warning: could not compute valid node positions for the given edge lengths.")
        print(f"scipy.optimize.minimize: {result.message}.")

    node_positions_as_array = result.x.reshape((-1, 2))
    node_positions_as_array = _fit_to_frame(node_positions_as_array, np.array(origin), np.array(scale), pad_by)
    node_positions = dict(zip(unique_nodes, node_positions_as_array))
    return node_positions


def _initialise_geometric_node_layout(edges, edge_length):
    """Initialises the node positions using the FR algorithm with weights.
    Shorter edges are given a larger weight such that the nodes experience a strong attractive force."""

    edge_weight = dict()
    for edge, length in edge_length.items():
        edge_weight[edge] = 1 / length
    node_positions = get_fruchterman_reingold_layout(edges, edge_weight=edge_weight)
    return np.array(list(node_positions.values()))


def _fit_to_frame(positions, origin, scale, pad_by):
    """Rotate, rescale and shift a set of positions such that they fit
    inside a frame while preserving the relative distances between
    them."""

    # find major axis
    delta = positions[np.newaxis, :] - positions[:, np.newaxis]
    distances = np.sum(delta**2, axis=-1)
    ii, jj = np.where(np.triu(distances)==np.max(distances))

    # use the first if there are several solutions
    ii = ii[0]
    jj = jj[0]

    # pivot around half-way point
    pivot = positions[ii] + 0.5 * delta[ii, jj]
    angle = _get_angle(*delta[ii, jj])

    if scale[0] < scale[1]: # portrait
        rotated_positions = _rotate((np.pi/2 - angle) % np.pi, positions, pivot)
    else: # landscape
        rotated_positions = _rotate(-angle % np.pi, positions, pivot)

    # shift to (0, 0)
    shifted_positions = rotated_positions - np.min(rotated_positions, axis=0)[np.newaxis, :]

    # rescale & center
    dx, dy = np.ptp(rotated_positions, axis=0)
    if dx/scale[0] < dy/scale[1]:
        rescaled_positions = shifted_positions * (1 - 2 * pad_by) * scale[1] / dy
        rescaled_positions[:, 0] += (scale[0] - np.ptp(rescaled_positions[:, 0])) / 2
        rescaled_positions[:, 1] += pad_by * scale[1]
    else:
        rescaled_positions = shifted_positions * (1 - 2 * pad_by) * scale[0] / dx
        rescaled_positions[:, 0] += pad_by * scale[0]
        rescaled_positions[:, 1] += (scale[1] - np.ptp(rescaled_positions[:, 1])) / 2

    # shift to origin
    reshifted_positions = rescaled_positions + np.array(origin)[np.newaxis, :]

    return reshifted_positions
