#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from uuid import uuid4
from scipy.spatial import cKDTree

from ._utils import (
    _get_unique_nodes,
    _get_angle,
    _get_interior_angle_between,
    _get_orthogonal_unit_vector,
    _get_point_along_spline,
    _get_tangent_at_point,
    _get_text_object_dimensions,
    _make_pretty,
    _rank,
    _get_n_points_on_a_circle,
)
from ._node_layout import (
    get_fruchterman_reingold_layout,
    get_random_layout,
    get_sugiyama_layout,
    get_circular_layout,
    get_community_layout,
    _reduce_node_overlap,
)
from ._edge_layout import (
    get_straight_edge_paths,
    _shift_edge,
    get_curved_edge_paths,
    get_bundled_edge_paths,
    get_selfloop_paths,
    _get_selfloop_path,
)
from ._artists import NodeArtist, EdgeArtist
from ._parser import parse_graph, _parse_edge_list, _is_directed
from ._deprecated import deprecated


BASE_SCALE = 1e-2
DEFAULT_COLOR = '#2c404c' # '#677e8c' # '#121f26' # '#23343f' # 'k',


@deprecated("Use Graph.draw() or InteractiveGraph.draw() instead.")
def draw(graph, node_positions=None, node_labels=None, edge_labels=None, edge_cmap='RdGy', ax=None, **kwargs):
    """
    Convenience function that tries to do "the right thing".

    For a full list of available arguments, and
    for finer control of the individual draw elements,
    please refer to the documentation of

        draw_nodes()
        draw_edges()
        draw_node_labels()
        draw_edge_labels()

    Arguments
    ----------
    graph: various formats
        Graph object to plot. Various input formats are supported.
        In order of precedence:
            - Edge list:
                Iterable of (source, target) or (source, target, weight) tuples,
                or equivalent (m, 2) or (m, 3) ndarray.
            - Adjacency matrix:
                Full-rank (n,n) ndarray, where n corresponds to the number of nodes.
                The absence of a connection is indicated by a zero.
            - igraph.Graph object
            - networkx.Graph object

    node_positions : dict node : (float, float)
        Mapping of nodes to (x, y) positions.
        If 'graph' is an adjacency matrix, nodes must be integers in range(n).

    node_labels : dict node : str (default None)
       Mapping of nodes to node labels.
       Only nodes in the dictionary are labelled.
       If 'graph' is an adjacency matrix, nodes must be integers in range(n).

    edge_labels : dict (source, target) : str (default None)
        Mapping of edges to edge labels.
        Only edges in the dictionary are labelled.

    ax : matplotlib.axis instance or None (default None)
       Axis to plot onto; if none specified, one will be instantiated with plt.gca().

    See Also
    --------
    draw_nodes()
    draw_edges()
    draw_node_labels()
    draw_edge_labels()

    TODO: return plot elements as dictionary

    """

    # Accept a variety of formats and convert to common denominator.
    nodes, edge_list, edge_weight = parse_graph(graph)
    is_directed = _is_directed(edge_list)

    if edge_weight:

        # If the graph is weighted, we want to visualise the weights using color.
        # Edge width is another popular choice when visualising weighted networks,
        # but if the variance in weights is large, this typically results in less
        # visually pleasing results.
        edge_color  = _get_color(edge_weight, cmap=edge_cmap)
        kwargs.setdefault('edge_color',  edge_color)

        # Plotting darker edges over lighter edges typically results in visually
        # more pleasing results. Here we hence specify the relative order in
        # which edges are plotted according to the color of the edge.
        edge_zorder = _get_zorder(edge_color)
        kwargs.setdefault('edge_zorder', edge_zorder)

    # Plot arrows if the graph has bi-directional edges.
    if is_directed:
        kwargs.setdefault('draw_arrows', True)
    else:
        kwargs.setdefault('draw_arrows', False)

    # Initialise node positions if none are given.
    if node_positions is None:
        node_positions = get_fruchterman_reingold_layout(edge_list, **kwargs)
    else:
        if set(node_positions.keys()).issuperset(_get_unique_nodes(edge_list)):
            # All node positions are given; nothing left to do.
            pass
        else:
            # Some node positions are given; however, either
            # 1) not all positions are provided, or
            # 2) there are some unconnected nodes in the graph.
            node_positions = get_fruchterman_reingold_layout(edge_list,
                                                             node_positions = node_positions,
                                                             fixed_nodes    = node_positions.keys(),
                                                             **kwargs)

    # Create axis if none is given.
    if ax is None:
        ax = plt.gca()

    # Draw plot elements.
    draw_edges(edge_list, node_positions, ax=ax, **kwargs)
    draw_nodes(node_positions, ax=ax, **kwargs)

    # This function needs to be called before any font sizes are adjusted.
    if 'node_size' in kwargs:
        _update_view(node_positions, ax=ax, node_size=kwargs['node_size'])
    else:
        _update_view(node_positions, ax=ax)

    if node_labels is not None:
        if not 'node_label_font_size' in kwargs:
            # set font size such that even the largest label fits inside node label face artist
            font_size = _get_font_size(ax, node_labels, **kwargs) * 0.9 # conservative fudge factor
            draw_node_labels(node_labels, node_positions, node_label_font_size=font_size, ax=ax, **kwargs)
        else:
            draw_node_labels(node_labels, node_positions, ax=ax, **kwargs)

    if edge_labels is not None:
        draw_edge_labels(edge_list, edge_labels, node_positions, ax=ax, **kwargs)

    # Improve default layout of axis.
    _make_pretty(ax)
    return ax


def _get_color(mydict, cmap='RdGy', vmin=None, vmax=None):
    """
    Map positive and negative floats to a diverging colormap,
    such that
        1) the midpoint of the colormap corresponds to a value of 0., and
        2) values above and below the midpoint are mapped linearly and in equal measure
           to increases in color intensity.

    Arguments:
    ----------
    mydict: dict key : float
        Mapping of graph element (node, edge) to a float.
        For example (source, target) : edge weight.

    cmap: str
        Matplotlib colormap specification.

    vmin, vmax: float
        Minimum and maximum float corresponding to the dynamic range of the colormap.

    Returns:
    --------
    newdict: dict key : (float, float, float, float)
        Mapping of graph element to RGBA tuple.

    """

    keys = mydict.keys()
    values = np.array(list(mydict.values()), dtype=np.float64)

    # apply vmin, vmax
    if vmin or vmax:
        values = np.clip(values, vmin, vmax)

    def abs(value):
        try:
            return np.abs(value)
        except TypeError as e: # value is probably None
            if isinstance(value, type(None)):
                return 0
            else:
                raise e

    # rescale values such that
    #  - the colormap midpoint is at zero-value, and
    #  - negative and positive values have comparable intensity values
    values /= np.nanmax([np.nanmax(np.abs(values)), abs(vmax), abs(vmin)]) # [-1, 1]
    values += 1. # [0, 2]
    values /= 2. # [0, 1]

    # convert value to color
    mapper = matplotlib.cm.ScalarMappable(cmap=cmap)
    mapper.set_clim(vmin=0., vmax=1.)
    colors = mapper.to_rgba(values)

    return {key: color for (key, color) in zip(keys, colors)}


def _get_zorder(color_dict):
    """
    Reorder plot elements such that darker items are plotted last and hence most prominent in the graph.
    This assumes that the background is white.
    """
    intensities = [rgba_to_grayscale(*v) for v in color_dict.values()]
    zorder = _rank(intensities)
    zorder = np.max(zorder) - zorder # reverse order as greater values correspond to lighter colors
    return {key: index for key, index in zip(color_dict.keys(), zorder)}


def rgba_to_grayscale(r, g, b, a=1):
    # https://stackoverflow.com/a/689547/2912349
    return (0.299 * r + 0.587 * g + 0.114 * b) * a


def _get_font_size(ax, node_labels, **kwargs):
    """
    Determine the maximum font size that results in labels that still all fit inside the node artist.

    TODO:
    -----
    - add font / fontfamily as optional argument
    - potentially, return a dictionary of font sizes instead; then rescale font sizes individually on a per node basis
    """

    # check if there are relevant parameters in kwargs:
    #   - node size,
    #   - edge width, or
    #   - node_label_font_size
    if 'node_size' in kwargs:
        node_size = kwargs['node_size']
    else:
        node_size = 3. # default

    if 'node_edge_width' in kwargs:
        node_edge_width = kwargs['node_edge_width']
    else:
        node_edge_width = 0.5 # default

    if 'node_label_font_size' in kwargs:
        node_label_font_size = kwargs['node_label_font_size']
    else:
        node_label_font_size = 12. # default

    # find widest node label; use its rescale factor to set font size for all labels
    widest = 0.
    rescale_factor = np.nan
    for key, label in node_labels.items():

        if isinstance(node_size, (int, float)):
            r = node_size
        elif isinstance(node_size, dict):
            r = node_size[key]

        if isinstance(node_edge_width, (int, float)):
            e = node_edge_width
        elif isinstance(node_edge_width, dict):
            e = node_edge_width[key]

        node_diameter = 2 * (r-e) * BASE_SCALE

        width, height = _get_text_object_dimensions(ax, label, size=node_label_font_size)

        if width > widest:
            widest = width
            rescale_factor = node_diameter / np.sqrt(width**2 + height**2)

    font_size = node_label_font_size * rescale_factor
    return font_size


@deprecated("Use Graph.draw_nodes() or InteractiveGraph.draw_nodes() instead.")
def draw_nodes(node_positions,
               node_shape='o',
               node_size=3.,
               node_edge_width=0.5,
               node_color='w',
               node_edge_color='k',
               node_alpha=1.0,
               ax=None,
               **kwargs):
    """
    Draw node markers at specified positions.

    Arguments
    ----------
    node_positions : dict node : (float, float)
        Mapping of nodes to (x, y) positions.

    node_shape : string or dict key : string (default 'o')
       The shape of the node. Specification is as for matplotlib.scatter
       marker, i.e. one of 'so^>v<dph8'.
       If a single string is provided all nodes will have the same shape.

    node_size : scalar or dict node : float (default 3.)
       Size (radius) of nodes.
       NOTE: Value is rescaled by BASE_SCALE (1e-2) to be compatible with layout routines in igraph and networkx.

    node_edge_width : scalar or dict key : float (default 0.5)
       Line width of node marker border.
       NOTE: Value is rescaled by BASE_SCALE (1e-2) to be compatible with layout routines in igraph and networkx.

    node_color : matplotlib color specification or dict node : color specification (default 'w')
       Node color.

    node_edge_color : matplotlib color specification or dict node : color specification (default 'k')
       Node edge color.

    node_alpha : scalar or dict node : float (default 1.)
       The node transparency.

    ax : matplotlib.axis instance or None (default None)
       Axis to plot onto; if none specified, one will be instantiated with plt.gca().

    Returns
    -------
    node_artists: dict node : artist
        Mapping of nodes to the node artists.

    """

    if ax is None:
        ax = plt.gca()

    # convert all inputs to dicts mapping node:property
    nodes = node_positions.keys()
    number_of_nodes = len(nodes)

    if isinstance(node_shape, str):
        node_shape = {node:node_shape for node in nodes}
    if isinstance(node_size, (int, float)):
        node_size = {node:node_size for node in nodes}
    if isinstance(node_edge_width, (int, float)):
        node_edge_width = {node: node_edge_width for node in nodes}
    if not isinstance(node_color, dict):
        node_color = {node:node_color for node in nodes}
    if not isinstance(node_edge_color, dict):
        node_edge_color = {node:node_edge_color for node in nodes}
    if isinstance(node_alpha, (int, float)):
        node_alpha = {node:node_alpha for node in nodes}

    # rescale
    node_size = {node: size  * BASE_SCALE for (node, size) in node_size.items()}
    node_edge_width = {node: width  * BASE_SCALE for (node, width) in node_edge_width.items()}

    artists = dict()
    for node in nodes:
        node_artist = NodeArtist(shape=node_shape[node],
                                 xy=node_positions[node],
                                 radius=node_size[node],
                                 facecolor=node_color[node],
                                 edgecolor=node_edge_color[node],
                                 linewidth=node_edge_width[node],
                                 alpha=node_alpha[node],
                                 zorder=2)

        # add artists to axis
        ax.add_artist(node_artist)

        # return handles to artists
        artists[node] = node_artist

    return artists


@deprecated("Use Graph.draw_edges() or InteractiveGraph.draw_edges() instead.")
def draw_edges(edge_list,
               node_positions,
               node_size=3.,
               edge_width=1.,
               edge_color='k',
               edge_alpha=1.,
               edge_zorder=None,
               draw_arrows=True,
               curved=False,
               ax=None,
               **kwargs):
    """
    Draw the edges of the network.

    Arguments
    ----------
    edge_list : m-long iterable of 2-tuples or equivalent (such as (m, 2) ndarray)
        List of edges. Each tuple corresponds to an edge defined by (source, target).

    node_positions : dict key : (float, float)
        Mapping of nodes to (x,y) positions

    node_size : scalar or (n,) or dict key : float (default 3.)
        Size (radius) of nodes.
        Used to offset edges when drawing arrow heads,
        such that the arrow heads are not occluded.
        If draw_nodes() and draw_edges() are called independently,
        make sure to set this variable to the same value.

    edge_width : float or dict (source, key) : width (default 1.)
        Line width of edges.
        NOTE: Value is rescaled by BASE_SCALE (1e-2) to be compatible with layout routines in igraph and networkx.

    edge_color : matplotlib color specification or dict (source, target) : color specification (default 'k')
       Edge color.

    edge_alpha : float or dict (source, target) : float (default 1.)
        The edge transparency,

    edge_zorder : int or dict (source, target) : int (default None)
        Order in which to plot the edges.
        If None, the edges will be plotted in the order they appear in 'adjacency'.
        Note: graphs typically appear more visually pleasing if darker coloured edges
        are plotted on top of lighter coloured edges.

    draw_arrows : bool, optional (default True)
        If True, draws edges with arrow heads.

    curved : bool, optional (default False)
        If True, draw edges as curved splines.

    ax : matplotlib.axis instance or None (default None)
       Axis to plot onto; if none specified, one will be instantiated with plt.gca().

    Returns
    -------
    artists: dict (source, target) : artist
        Mapping of edges to EdgeArtists.

    """

    if ax is None:
        ax = plt.gca()

    edge_list = _parse_edge_list(edge_list)
    nodes = node_positions.keys()

    # convert node and edge to dictionaries if they are not dictionaries already;
    # if dictionaries are provided, make sure that they are complete;
    # fill any missing entries with the default values
    if not isinstance(node_size, dict):
        node_size = {node:node_size for node in nodes}
    else:
        for node in nodes:
            node_size.setdefault(node, 3.)

    if not isinstance(edge_width, dict):
        edge_width = {edge: edge_width for edge in edge_list}
    else:
        for edge in edge_list:
            edge_width.setdefault(edge, 1.)

    if not isinstance(edge_color, dict):
        edge_color = {edge: edge_color for edge in edge_list}
    else:
        for edge in edge_list:
            edge_color.setdefault(edge, 'k')

    if not isinstance(edge_alpha, dict):
        edge_alpha = {edge: edge_alpha for edge in edge_list}
    else:
        for edge in edge_list:
            edge_alpha.setdefault(edge, 1.)

    # rescale
    node_size  = {node: size  * BASE_SCALE  for (node, size)  in node_size.items()}
    edge_width = {edge: width * BASE_SCALE for (edge, width) in edge_width.items()}

    # order edges if necessary
    if edge_zorder:
        for edge in edge_list:
            edge_zorder.setdefault(edge, max(edge_zorder.values()))
        edge_list = sorted(edge_zorder, key=lambda k: edge_zorder[k])

    # compute edge paths
    if not curved:
        edge_paths = get_straight_edge_paths(edge_list, node_positions, edge_width)
    else:
        edge_paths = get_curved_edge_paths(edge_list, node_positions)

    # NOTE: At the moment, only the relative zorder is honored, not the absolute value.
    artists = dict()
    for (source, target) in edge_list:

        width = edge_width[(source, target)]
        color = edge_color[(source, target)]
        alpha = edge_alpha[(source, target)]
        offset = node_size[target]

        if ((target, source) in edge_list) and not curved: # i.e. bidirectional, straight edges
            # plot half arrow / line
            shape = 'right'
        else:
            shape = 'full'

        if draw_arrows:
            head_length = 2 * width
            head_width = 3 * width
        else:
            head_length = 1e-10 # 0 throws error
            head_width = 1e-10 # 0 throws error

        edge_artist = EdgeArtist(
            midline     = edge_paths[(source, target)],
            width       = width,
            facecolor   = color,
            alpha       = alpha,
            head_length = head_length,
            head_width  = head_width,
            zorder      = 1,
            edgecolor   = 'none',
            linewidth   = 0.,
            offset      = offset,
            shape       = shape,
            curved      = curved,
        )
        ax.add_artist(edge_artist)
        artists[(source, target)] = edge_artist

    return artists


@deprecated("Use Graph.draw_node_labels() or InteractiveGraph.draw_node_labels() instead.")
def draw_node_labels(node_labels,
                     node_positions,
                     node_label_font_size=12,
                     node_label_font_color='k',
                     node_label_font_family='sans-serif',
                     node_label_font_weight='normal',
                     node_label_font_alpha=1.,
                     node_label_bbox=dict(alpha=0.),
                     node_label_horizontalalignment='center',
                     node_label_verticalalignment='center',
                     node_label_offset=(0., 0.),
                     clip_on=False,
                     ax=None,
                     **kwargs):
    """
    Draw node labels.

    Arguments
    ---------
    node_positions : dict node : (float, float)
        Mapping of nodes to (x, y) positions

    node_labels : dict key : str
       Mapping of nodes to labels.
       Only nodes in the dictionary are labelled.

    node_label_font_size : int (default 12)
       Font size for text labels

    node_label_font_color : str (default 'k')
       Font color string

    node_label_font_family : str (default='sans-serif')
       Font family

    node_label_font_weight : str (default='normal')
       Font weight

    node_label_font_alpha : float (default 1.)
       Text transparency

    node_label_bbox : matplotlib bbox instance (default {'alpha': 0})
       Specify text box shape and colors.

    node_label_horizontalalignment: str
        Horizontal label alignment inside bbox.

    node_label_verticalalignment: str
        Vertical label alignment inside bbox.

    node_label_offset: 2-tuple or equivalent iterable (default (0.,0.))
        (x, y) offset from node centre of label position.

    clip_on : bool (default False)
       Turn on clipping at axis boundaries.

    ax : matplotlib.axis instance or None (default None)
       Axis to plot onto; if none specified, one will be instantiated with plt.gca().

    Returns
    -------
    artists: dict
        Dictionary mapping node indices to text objects.

    @reference
    Borrowed with minor modifications from networkx/drawing/nx_pylab.py

    """

    if ax is None:
        ax = plt.gca()

    dx, dy = node_label_offset

    artists = dict()  # there is no text collection so we'll fake one
    for node, label in node_labels.items():
        try:
            x, y = node_positions[node]
        except KeyError:
            print("Cannot draw node label for node with ID {}. The node has no position assigned to it.".format(node))
            continue
        x += dx
        y += dy
        text_object = ax.text(x, y,
                              label,
                              size=node_label_font_size,
                              color=node_label_font_color,
                              alpha=node_label_font_alpha,
                              family=node_label_font_family,
                              weight=node_label_font_weight,
                              bbox=node_label_bbox,
                              horizontalalignment=node_label_horizontalalignment,
                              verticalalignment=node_label_verticalalignment,
                              transform=ax.transData,
                              clip_on=clip_on)
        artists[node] = text_object

    return artists


@deprecated("Use Graph.draw_edge_labels() or InteractiveGraph.draw_edge_labels() instead.")
def draw_edge_labels(edge_list,
                     edge_labels,
                     node_positions,
                     edge_label_position=0.5,
                     edge_label_font_size=10,
                     edge_label_font_color='k',
                     edge_label_font_family='sans-serif',
                     edge_label_font_weight='normal',
                     edge_label_font_alpha=1.,
                     edge_label_bbox=None,
                     edge_label_horizontalalignment='center',
                     edge_label_verticalalignment='center',
                     clip_on=False,
                     edge_width=1.,
                     ax=None,
                     rotate=True,
                     edge_label_zorder=10000,
                     **kwargs):
    """
    Draw edge labels.

    Arguments
    ---------

    edge_list: m-long list of 2-tuples
        List of edges. Each tuple corresponds to an edge defined by (source, target).

    edge_labels : dict (source, target) : str
        Mapping of edges to edge labels.
        Only edges in the dictionary are labelled.

    node_positions : dict node : (float, float)
        Mapping of nodes to (x, y) positions

    edge_label_position : float
        Relative position along the edge where the label is placed.
            head   : 0.
            centre : 0.5 (default)
            tail   : 1.

    edge_label_font_size : int (default 12)
       Font size

    edge_label_font_color : str (default 'k')
       Font color

    edge_label_font_family : str (default='sans-serif')
       Font family

    edge_label_font_weight : str (default='normal')
       Font weight

    edge_label_font_alpha : float (default 1.)
       Text transparency

    edge_label_bbox : Matplotlib bbox
       Specify text box shape and colors.

    edge_label_horizontalalignment: str
        Horizontal label alignment inside bbox.

    edge_label_verticalalignment: str
        Vertical label alignment inside bbox.

    clip_on : bool (default=False)
       Turn on clipping at axis boundaries.

    edge_label_zorder : int (default 10000)
        Set the zorder of edge labels.
        Choose a large number to ensure that the labels are plotted on top of the edges.

    edge_width : float or dict (source, key) : width (default 1.)
        Line width of edges.
        NOTE: Value is rescaled by BASE_SCALE (1e-2) to be compatible with layout routines in igraph and networkx.

    ax : matplotlib.axis instance or None (default None)
       Axis to plot onto; if none specified, one will be instantiated with plt.gca().

    Returns
    -------
    artists: dict (source, target) : text object
        Mapping of edges to edge label artists.

    @reference
    Borrowed with minor modifications from networkx/drawing/nx_pylab.py

    """

    if ax is None:
        ax = plt.gca()

    if isinstance(edge_width, (int, float)):
        edge_width = {edge: edge_width for edge in edge_list}

    edge_width = {edge: width * BASE_SCALE for (edge, width) in edge_width.items()}

    text_items = {}
    for (n1, n2), label in edge_labels.items():

        if n1 != n2:

            (x1, y1) = node_positions[n1]
            (x2, y2) = node_positions[n2]

            if (n2, n1) in edge_list: # i.e. bidirectional edge --> need to shift label to stay on edge
                x1, y1, x2, y2 = _shift_edge(x1, y1, x2, y2, delta=1.*edge_width[(n1, n2)])

            (x, y) = (x1 * edge_label_position + x2 * (1.0 - edge_label_position),
                      y1 * edge_label_position + y2 * (1.0 - edge_label_position))

            if rotate:
                ange += _get_angle(x2-x1, y2-y1, radians=True)
                # make label orientation "right-side-up"
                if angle > 90:
                    angle -= 180
                if angle < - 90:
                    angle += 180
                # transform data coordinate angle to screen coordinate angle
                xy = np.array((x, y))
                trans_angle = ax.transData.transform_angles(np.array((angle,)),
                                                            xy.reshape((1, 2)))[0]
            else:
                trans_angle = 0.0

            if edge_label_bbox is None: # use default box of white with white border
                edge_label_bbox = dict(boxstyle='round',
                                       ec=(1.0, 1.0, 1.0),
                                       fc=(1.0, 1.0, 1.0))

            t = ax.text(x, y,
                        label,
                        size=edge_label_font_size,
                        color=edge_label_font_color,
                        alpha=edge_label_font_alpha,
                        family=edge_label_font_family,
                        weight=edge_label_font_weight,
                        bbox=edge_label_bbox,
                        horizontalalignment=edge_label_horizontalalignment,
                        verticalalignment=edge_label_verticalalignment,
                        rotation=trans_angle,
                        transform=ax.transData,
                        zorder=edge_label_zorder,
                        clip_on=clip_on,
                        )
            text_items[(n1, n2)] = t

        else: # n1 == n2, i.e. a self-loop
            warnings.warn("Plotting of edge labels for self-loops not supported. Ignoring edge with label: {}".format(label))

    return text_items


def _update_view(node_positions, ax, node_size=3.):
    # Pad x and y limits as patches are not registered properly
    # when matplotlib sets axis limits automatically.
    # Hence we need to set them manually.

    if isinstance(node_size, dict):
        maxs = np.max(list(node_size.values())) * BASE_SCALE
    else:
        maxs = node_size * BASE_SCALE

    maxx, maxy = np.max(list(node_positions.values()), axis=0)
    minx, miny = np.min(list(node_positions.values()), axis=0)

    w = maxx-minx
    h = maxy-miny
    padx, pady = 0.05*w + maxs, 0.05*h + maxs
    corners = (minx-padx, miny-pady), (maxx+padx, maxy+pady)

    ax.update_datalim(corners)
    ax.autoscale_view()
    ax.get_figure().canvas.draw()


# --------------------------------------------------------------------------------
# interactive plotting


def _add_doc(value):
    def _doc(func):
        func.__doc__ = value
        return func
    return _doc


class BaseGraph(object):
    """The Graph base class.

    Arguments
    ----------
    edges : list of (source node, target node) 2-tuples
        List of edges.

    nodes : list of node IDs or None (default None)
        If None, `nodes` is initialised to the set of the flattened `edges`.

    node_layout : str or dict node : (float x, float y) (default 'spring')
        If node_layout is a string, the node positions are computed using
        the indicated method:
        - 'random'    : place nodes in random positions;
        - 'circular'  : place nodes regularly spaced on a circle;
        - 'spring'    : place nodes using a force-directed layout (Fruchterman-Reingold algorithm);
        - 'dot'       : place nodes using the Sugiyama algorithm; the graph should be directed and acyclic;
        - 'community' : place nodes such that nodes belonging to the same community are grouped together
        If node_layout is a dict, keys are nodes and values are (x, y) positions.

    node_layout_kwargs : dict (default None)
        Keyword arguments passed to node layout functions.
        See the documentation of the following functions for a full description
        of available options:
        - get_random_layout
        - get_circular_layout
        - get_fruchterman_reingold_layout
        - get_sugiyama_layout
        - get_community_layout

    node_shape : string or dict node : string (default 'o')
       The shape of the node. Specification is as for matplotlib.scatter
       marker, i.e. one of 'so^>v<dph8'.
       If a single string is provided all nodes will have the same shape.

    node_size : scalar or dict node : float (default 3.)
       Size (radius) of nodes.
       NOTE: Value is rescaled by BASE_SCALE (1e-2) to be compatible with layout routines in igraph and networkx.

    node_edge_width : scalar or dict node : float (default 0.5)
       Line width of node marker border.
       NOTE: Value is rescaled by BASE_SCALE (1e-2) to be compatible with layout routines in igraph and networkx.

    node_color : matplotlib color specification or dict node : color (default 'w')
       Node color.

    node_edge_color : matplotlib color specification or dict node : color (default DEFAULT_COLOR)
       Node edge color.

    node_alpha : scalar or dict node : float (default 1.)
       The node transparency.

    node_zorder : scalar or dict node : float (default 2)
       Order in which to plot the nodes.

    node_labels : bool or dict node : str (default False)
       If True, the nodes are labelled with their node IDs.
       If the node labels are supposed to be distinct from the node IDs,
       supply a dictionary mapping nodes to node labels.
       Only nodes in the dictionary are labelled.

    node_label_offset: scalar or 2-tuple or equivalent iterable (default (0., 0.))
        A (dx, dy) tuple specifies the exact offset from the node position.
        If a scalar delta is specified, the value is interpreted as a distance, and
        the label is placed delta away from the node position while trying to
        reduce node/label, node/edge, and label/label overlaps.

    node_label_fontdict : dict
        Keyword arguments passed to matplotlib.text.Text.
        For a full list of available arguments see the matplotlib documentation.
        The following default values differ from the defaults for matplotlib.text.Text:
            - size (adjusted to fit into node artists if offset is (0, 0))
            - horizontalalignment (default here: 'center')
            - verticalalignment (default here: 'center')
            - clip_on (default here: False)
            - zorder (default here: inf)

    edge_width : float or dict (source, target) : width (default 1.)
        Line width of edges.
        NOTE: Value is rescaled by BASE_SCALE (1e-2) to be compatible with layout routines in igraph and networkx.

    edge_color : matplotlib color specification or dict (source, target) : color (default DEFAULT_COLOR)
       Edge color.

    edge_alpha : float or dict (source, target) : float (default 1.)
        The edge transparency,

    edge_zorder : int or dict (source, target) : int (default 1)
        Order in which to plot the edges.
        If None, the edges will be plotted in the order they appear in 'adjacency'.
        Note: graphs typically appear more visually pleasing if darker edges
        are plotted on top of lighter edges.

    arrows : bool, optional (default False)
        If True, draw edges with arrow heads.

    edge_layout : str or dict edge : segments (default 'straight')
        If edge_layout is a string, determine the layout internally:
        - 'straight' : draw edges as straight lines
        - 'curved'   : draw edges as curved splines; the spline control points are optimised to avoid other nodes and edges
        - 'bundled'  : draw edges as edge bundles
        If edge_layout is a dict, the keys are edges and the
        values are edge paths in the form iterables of (x, y)
        tuples, the edge segments.

    edge_layout_kwargs : dict (default None)
        Keyword arguments passed to edge layout functions.
        See the documentation of the following functions for a full description
        of available options:
        - get_straight_edge_paths
        - get_curved_edge_paths
        - get_bundled_edge_paths

    edge_labels : bool or dict edge : str
        If True, the edges are labelled with their edge IDs.
        If the edge labels are supposed to be distinct from the edge IDs,
        supply a dictionary mapping edges to edge labels.
        Only edges in the dictionary are labelled.

    edge_label_position : float
        Relative position along the edge where the label is placed.
            head   : 0.
            centre : 0.5 (default)
            tail   : 1.

    edge_label_rotate : bool (default True)
        If True, edge labels are rotated such that they track the orientation of their edges.
        If False, edge labels are not rotated; the angle of the text is parallel to the axis.

    edge_label_fontdict : dict
        Keyword arguments passed to matplotlib.text.Text.
        For a full list of available arguments see the matplotlib documentation.
        The following default values differ from the defaults for matplotlib.text.Text:
            - horizontalalignment (default here: 'center'),
            - verticalalignment (default here: 'center')
            - clip_on (default here: False),
            - bbox (default here: dict(boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)),
            - zorder (default here: inf),
            - rotation (determined by edge_label_rotate argument)

    origin : (float x, float y) tuple or None (default (0, 0))
        The lower left hand corner of the bounding box specifying the extent of the canvas.

    scale : (float delta x, float delta y) or None (default (1, 1))
        The width and height of the bounding box specifying the extent of the canvas.

    prettify : bool (default True)
        If True, despine and remove ticks and tick labels.
        Set figure background to white. Set axis aspect to equal.

    ax : matplotlib.axis instance or None (default None)
       Axis to plot onto; if none specified, one will be instantiated with plt.gca().


    Attributes:
    -----------
    node_artists : dict node : NodeArtist instance
        The node artists.

    edge_artists : dict edge : EdgeArtist instance
        The edge artists.

    node_label_artists : dict node : matplotlib.Text
        The node label text objects (if applicable).

    edge_label_artists : dict edge : matplotlib.Text
        The edge label text objects (if applicable).

    node_positions : dict node : (x, y) tuple
        The node positions.

    """

    def __init__(self, edges,
                 nodes=None,
                 node_layout='spring',
                 node_layout_kwargs=None,
                 node_shape='o',
                 node_size=3.,
                 node_edge_width=0.5,
                 node_color='w',
                 node_edge_color=DEFAULT_COLOR,
                 node_alpha=1.0,
                 node_zorder=2,
                 node_labels=False,
                 node_label_offset=(0., 0.),
                 node_label_fontdict=None,
                 edge_width=1.,
                 edge_color=DEFAULT_COLOR,
                 edge_alpha=0.5,
                 edge_zorder=1,
                 arrows=False,
                 edge_layout='straight',
                 edge_layout_kwargs=None,
                 edge_labels=False,
                 edge_label_position=0.5,
                 edge_label_rotate=True,
                 edge_label_fontdict=None,
                 origin=(0., 0.),
                 scale=(1., 1.),
                 prettify=True,
                 ax=None,
                 *args, **kwargs
    ):
        self.edges = _parse_edge_list(edges)

        self.nodes = self._initialize_nodes(nodes)

        # Convert all node and edge parameters to dictionaries.
        node_shape      = self._normalize_string_argument(node_shape, self.nodes, 'node_shape')
        node_size       = self._normalize_numeric_argument(node_size, self.nodes, 'node_size')
        node_edge_width = self._normalize_numeric_argument(node_edge_width, self.nodes, 'node_edge_width')
        node_color      = self._normalize_color_argument(node_color, self.nodes, 'node_color')
        node_edge_color = self._normalize_color_argument(node_edge_color, self.nodes, 'node_edge_color')
        node_alpha      = self._normalize_numeric_argument(node_alpha, self.nodes, 'node_alpha')
        node_zorder     = self._normalize_numeric_argument(node_zorder, self.nodes, 'node_zorder')
        edge_width      = self._normalize_numeric_argument(edge_width, self.edges, 'edge_width')
        edge_color      = self._normalize_color_argument(edge_color, self.edges, 'edge_color')
        edge_alpha      = self._normalize_numeric_argument(edge_alpha, self.edges, 'edge_alpha')
        edge_zorder     = self._normalize_numeric_argument(edge_zorder, self.edges, 'edge_zorder')

        # Rescale.
        node_size = self._rescale(node_size, BASE_SCALE)
        node_edge_width = self._rescale(node_edge_width, BASE_SCALE)
        edge_width = self._rescale(edge_width, BASE_SCALE)

        # Initialise node and edge layouts.
        self.node_positions = self._initialize_node_layout(
            node_layout, node_layout_kwargs, origin, scale, node_size)

        self.edge_paths, self.edge_layout, self.edge_layout_kwargs = self._initialize_edge_layout(
            edge_layout, edge_layout_kwargs, origin, scale, edge_width)

        # Draw plot elements
        self.ax = self._initialize_axis(ax)

        self.edge_artists = dict()
        self.draw_edges(self.edge_paths, edge_width, edge_color, edge_alpha,
                        edge_zorder, arrows, node_size)

        self.node_artists = dict()
        self.draw_nodes(self.nodes, self.node_positions,
                        node_shape, node_size, node_edge_width,
                        node_color, node_edge_color, node_alpha, node_zorder)

        # This function needs to be called before any font sizes are adjusted,
        # as the axis dimensions affect the effective font size.
        self._update_view()

        if node_labels:
            if isinstance(node_labels, bool):
                node_labels = dict(zip(self.nodes, self.nodes))
            self.node_label_fontdict = self._initialize_node_label_fontdict(
                node_label_fontdict, node_labels, node_label_offset)
            self.node_label_offset, self._recompute_node_label_offsets =\
                self._initialize_node_label_offset(node_labels, node_label_offset)
            if self._recompute_node_label_offsets:
                self._update_node_label_offsets()
            self.node_label_artists = dict()
            self.draw_node_labels(node_labels, self.node_label_fontdict)

        if edge_labels:
            if isinstance(edge_labels, bool):
                edge_labels = dict(zip(self.edges, self.edges))
            self.edge_label_fontdict = self._initialize_edge_label_fontdict(edge_label_fontdict)
            self.edge_label_position = edge_label_position
            self.edge_label_rotate = edge_label_rotate
            self.edge_label_artists = dict()
            self.draw_edge_labels(edge_labels, self.edge_label_position,
                                  self.edge_label_rotate, self.edge_label_fontdict)

        if prettify:
            _make_pretty(self.ax)


    def _initialize_nodes(self, nodes):
        nodes_in_edges = _get_unique_nodes(self.edges)
        if nodes is None:
            return nodes_in_edges
        else:
            if set(nodes).issuperset(nodes_in_edges):
                return nodes
            else:
                msg = "There are some node IDs in the edgelist not present in `nodes`. "
                msg += "`nodes` has to be the superset of `edges`."
                msg += "\nThe following nodes are missing:"
                missing = set(nodes_in_edges) - set(nodes)
                for node in missing:
                    msg += f"\n\t{node}"
                raise ValueError(msg)


    def _normalize_numeric_argument(self, numeric_or_dict, dict_keys, variable_name):
        if isinstance(numeric_or_dict, (int, float, np.integer, np.float)):
            return {key : numeric_or_dict for key in dict_keys}
        elif isinstance(numeric_or_dict, dict):
            self._check_completeness(numeric_or_dict, dict_keys, variable_name)
            self._check_types(numeric_or_dict.values(), (int, float, np.integer, np.float), variable_name)
            return numeric_or_dict
        else:
            msg = f"The type of {variable_name} has to be either a int, float, or a dict."
            msg += f"\nThe current type is {type(numeric_or_dict)}."
            raise TypeError(msg)


    def _check_completeness(self, given_set, desired_set, variable_name):
        # ensure that iterables are sets
        # TODO: check that iterables can safely be converted to sets (unlike dict keys)
        given_set = set(given_set)
        desired_set = set(desired_set)

        complete = given_set.issuperset(desired_set)
        if not complete:
            missing = desired_set - given_set
            msg = f"{variable_name} is incomplete. The following elements are missing:"
            for item in missing:
                msg += f"\n-{item}"
            raise ValueError(msg)


    def _check_types(self, items, types, variable_name):
        for item in items:
            if not isinstance(item, types):
                msg = f"Item {item} in {variable_name} is of the wrong type."
                msg += f"\nExpected type: {types}"
                msg += f"\nActual type: {type(item)}"
                raise TypeError(msg)


    def _normalize_string_argument(self, str_or_dict, dict_keys, variable_name):
        if isinstance(str_or_dict, str):
            return {key : str_or_dict for key in dict_keys}
        elif isinstance(str_or_dict, dict):
            self._check_completeness(set(str_or_dict), dict_keys, variable_name)
            self._check_types(str_or_dict.values(), str, variable_name)
            return str_or_dict
        else:
            msg = f"The type of {variable_name} has to be either a str or a dict."
            msg += f"The current type is {type(str_or_dict)}."
            raise TypeError(msg)


    def _normalize_color_argument(self, color_or_dict, dict_keys, variable_name):
        if matplotlib.colors.is_color_like(color_or_dict):
            return {key : color_or_dict for key in dict_keys}
        elif color_or_dict is None:
            return {key : color_or_dict for key in dict_keys}
        elif isinstance(color_or_dict, dict):
            self._check_completeness(set(color_or_dict), dict_keys, variable_name)
            # TODO: assert that each element is a valid color
            return color_or_dict
        else:
            msg = f"The type of {variable_name} has to be either a valid matplotlib color specification or a dict."
            raise TypeError(msg)


    def _rescale(self, mydict, scalar):
        return {key: value * scalar for (key, value) in mydict.items()}


    def _initialize_node_layout(self, node_layout, node_layout_kwargs, origin, scale, node_size):
        if node_layout_kwargs is None:
            node_layout_kwargs = dict()

        if isinstance(node_layout, str):
            if (node_layout == 'spring') or (node_layout == 'dot'):
                node_layout_kwargs.setdefault('node_size', node_size)
            return self._get_node_positions(node_layout, node_layout_kwargs, origin, scale)

        elif isinstance(node_layout, dict):
            self._check_completeness(set(node_layout), set(self.nodes), 'node_layout')
            return node_layout


    def _get_node_positions(self, node_layout, node_layout_kwargs, origin, scale):
        if node_layout == 'spring':
            node_positions = get_fruchterman_reingold_layout(
                self.edges, nodes=self.nodes, origin=origin, scale=scale, **node_layout_kwargs)
            if len(node_positions) > 3: # Qhull fails for 2 or less nodes
                node_positions = _reduce_node_overlap(node_positions, origin, scale)
            return node_positions
        if node_layout == 'community':
            node_positions = get_community_layout(
                self.edges, origin=origin, scale=scale, **node_layout_kwargs)
            if len(node_positions) > 3: # Qhull fails for 2 or less nodes
                node_positions = _reduce_node_overlap(node_positions, origin, scale)
            return node_positions
        elif node_layout == 'circular':
            return get_circular_layout(
                self.edges, nodes=self.nodes, origin=origin, scale=scale, **node_layout_kwargs)
        elif node_layout == 'dot':
            return get_sugiyama_layout(
                self.edges, nodes=self.nodes, origin=origin, scale=scale, **node_layout_kwargs)
        elif node_layout == 'random':
            return get_random_layout(
                self.edges, nodes=self.nodes, origin=origin, scale=scale, **node_layout_kwargs)
        else:
            implemented = ['spring', 'dot', 'random', 'circular']
            msg = f"Node layout {node_layout} not implemented. Available layouts are:"
            for method in implemented:
                msg += f"\n\t{method}"
            raise NotImplementedError(msg)


    def _initialize_edge_layout(self, edge_layout, edge_layout_kwargs, origin, scale, edge_width):
        if edge_layout_kwargs is None:
            edge_layout_kwargs = dict()

        if edge_layout == "straight":
            edge_layout_kwargs.setdefault('edge_width', edge_width)
            edge_layout_kwargs.setdefault('origin', origin)
            edge_layout_kwargs.setdefault('scale', scale)
            edge_layout_kwargs.setdefault('selfloop_radius', 0.05 * np.linalg.norm(scale))
        elif edge_layout == 'curved':
            edge_layout_kwargs.setdefault('origin', origin)
            edge_layout_kwargs.setdefault('scale', scale)
            edge_layout_kwargs.setdefault('selfloop_radius', 0.05 * np.linalg.norm(scale))
            area = np.product(scale)
            k = np.sqrt(area / float(len(self.nodes))) # expected distance between nodes
            # As there are multiple control points per edge,
            # edge segments should be much shorter. k hence needs to be smaller.
            k *= 0.1
            edge_layout_kwargs.setdefault('k', k)
        elif edge_layout == 'bundled':
            edge_layout_kwargs.setdefault('k', 500)
            edge_layout_kwargs.setdefault('total_cycles', 6)

        if isinstance(edge_layout, str):
            edge_paths = self._get_edge_paths(self.edges, self.node_positions,
                                                   edge_layout, edge_layout_kwargs)
        elif isinstance(edge_layout, dict):
            self._check_completeness(edge_layout, self.edges, 'edge_layout')
            edge_paths = edge_layout

            # determine a sensible edge_layout in case node positions change
            path_lengths = np.array([len(path) for path in edge_paths.values()])
            if np.any(path_lengths) > 2:
                edge_layout = 'curved'
            else:
                edge_layout = 'straight'
        else:
            raise TypeError("Variable `edge_layout` either a string or a dict mapping edges to edge paths.")

        return edge_paths, edge_layout, edge_layout_kwargs


    def _initialize_axis(self, ax):
        if ax is None:
            return plt.gca()
        elif isinstance(ax, matplotlib.axes._subplots.Axes):
            return ax
        else:
            raise TypeError(f"Variable 'ax' either None or a matplotlib axis instance. However, type(ax) is {type(ax)}.")


    def draw_nodes(self, nodes, node_positions, node_shape, node_size,
                   node_edge_width, node_color, node_edge_color, node_alpha,
                   node_zorder):
        """
        Draw node markers at specified positions.

        Arguments
        ----------
        nodes : list
            List of node IDs.

        node_positions : dict node : (float, float)
            Mapping of nodes to (x, y) positions.

        node_shape : dict node : string
           The shape of the node. Specification is as for matplotlib.scatter
           marker, i.e. one of 'so^>v<dph8'.

        node_size : dict node : float
           Size (radius) of nodes.

        node_edge_width : dict node : float
           Line width of node marker border.

        node_color : dict node : matplotlib color specification
           Node color.

        node_edge_color : dict node : matplotlib color specification
           Node edge color.

        node_alpha : dict node : float
           The node transparency.

        node_zorder : dict node : int
            The z-order of nodes.

        Updates
        -------
        node_artists: dict node : artist
            Mapping of nodes to the node artists.

        """
        for node in nodes:
            node_artist = NodeArtist(shape=node_shape[node],
                                     xy=node_positions[node],
                                     radius=node_size[node],
                                     facecolor=node_color[node],
                                     edgecolor=node_edge_color[node],
                                     linewidth=node_edge_width[node],
                                     alpha=node_alpha[node],
                                     zorder=node_zorder[node])
            self.ax.add_patch(node_artist)

            if node in self.node_artists:
                self.node_artists[node].remove()
            self.node_artists[node] = node_artist


    def _update_node_artists(self, nodes):
        for node in nodes:
            self.node_artists[node].xy = self.node_positions[node]


    def _get_edge_paths(self, edges, node_positions, edge_layout, edge_layout_kwargs):
        """
        Arguments
        ----------
        edges : m-long iterable of 2-tuples or equivalent (such as (m, 2) ndarray)
            List of edges. Each tuple corresponds to an edge defined by (source, target).

        node_positions : dict node : (float, float)
            Mapping of nodes to (x,y) positions

        edge_layout : 'straight', 'curved' or 'bundled' (default 'straight')
            If 'straight', draw edges as straight lines.
            If 'curved', draw edges as curved splines. The spline control points
            are optimised to avoid other nodes and edges.
            If 'bundled', draw edges as edge bundles.

        edge_layout_kwargs : dict
            Keyword arguments passed to edge layout functions.
            See the documentation of the following functions for a full list of
            available options:
            - get_straight_edge_paths
            - get_curved_edge_paths
            - get_bundled_edge_paths

        Returns:
        --------
        edge_paths : dict edge : path
            Dictionary mapping edges to arrays of (x, y) tuples, the edge segments.

        """

        if edge_layout == 'straight':
            edge_paths = get_straight_edge_paths(edges, node_positions, edge_layout_kwargs['edge_width'])
            selfloop_paths = get_selfloop_paths(
                edges, node_positions, edge_layout_kwargs['selfloop_radius'], edge_layout_kwargs['origin'], edge_layout_kwargs['scale'])
            edge_paths.update(selfloop_paths)
        elif edge_layout == 'curved':
            edge_paths = get_curved_edge_paths(edges, node_positions, **edge_layout_kwargs)
        elif edge_layout == 'bundled':
            edge_paths = get_bundled_edge_paths(edges, node_positions, **edge_layout_kwargs)
        else:
            raise NotImplementedError(f"Variable edge_layout one of 'straight', 'curved' or 'bundled', not {edge_layout}")

        return edge_paths


    def draw_edges(self, edge_path, edge_width, edge_color, edge_alpha,
                   edge_zorder, arrows, node_size):
        """
        Draw the edges of the network.

        Arguments
        ----------
        edge_path : dict edge : path
            Dictionary mapping edges to arrays of (x, y) tuples, the edge segments.

        edge_width : dict edge : width
            Line width of edges.

        edge_color :  dict edge : matplotlib color specification
           Edge color.

        edge_alpha : dict edge : float
            The edge transparency,

        edge_zorder : dict edge : int
            Order in which to plot the edges.
            Hint: graphs typically appear more visually pleasing if darker edges
            are plotted on top of lighter edges.

        arrows : bool
            If True, draws edges with arrow heads.

        node_size : dict node : float
            Size (radius) of nodes. Required to offset edges from nodes.

        Updates
        -------
        self.edge_artists: dict (source, target) : artist
            Mapping of edges to EdgeArtists.

        """

        for edge in edge_path:

            curved = False if (len(edge_path[edge]) == 2) else True

            source, target = edge
            if ((target, source) in edge_path) and not curved: # i.e. bidirectional, straight edges
                shape = 'right' # i.e. plot half arrow / thin line shifted to the right
            else:
                shape = 'full'

            if arrows:
                head_length = 2 * edge_width[edge]
                head_width = 3 * edge_width[edge]
            else:
                head_length = 1e-10 # 0 throws error
                head_width = 1e-10 # 0 throws error

            edge_artist = EdgeArtist(
                midline     = edge_path[edge],
                width       = edge_width[edge],
                facecolor   = edge_color[edge],
                alpha       = edge_alpha[edge],
                head_length = head_length,
                head_width  = head_width,
                edgecolor   = 'none',
                linewidth   = 0.,
                offset      = node_size[target],
                shape       = shape,
                curved      = curved,
                zorder      = edge_zorder[edge],
            )
            self.ax.add_patch(edge_artist)

            if edge in self.edge_artists:
                self.edge_artists[edge].remove()
            self.edge_artists[edge] = edge_artist


    def _update_edge_artists(self, edge_paths=None):
        if edge_paths is None:
            edge_paths = self.edge_paths

        for edge, path in edge_paths.items():
            self.edge_artists[edge].update_midline(path)
            self.ax.draw_artist(self.edge_artists[edge])


    def _update_edges(self, edges):
        edge_paths = dict()
        if self.edge_layout == 'straight':
            edge_paths.update(self._update_straight_edge_paths([(source, target) for (source, target) in edges if source != target]))
            edge_paths.update(self._update_selfloop_paths([(source, target) for (source, target) in edges if source == target]))
        elif self.edge_layout == 'curved':
            edge_paths.update(self._update_curved_edge_paths(edges))
        elif self.edge_layout == 'bundled':
            edge_paths.update(self._update_bundled_edge_paths(edges))
        self.edge_paths.update(edge_paths)
        self._update_edge_artists(edge_paths)


    def _update_straight_edge_paths(self, edges):
        # remove self-loops
        edges = [(source, target) for source, target in edges if source != target]

        edge_paths = dict()
        for (source, target) in edges:
            x0, y0 = self.node_positions[source]
            x1, y1 = self.node_positions[target]

            # # shift edge right if bi-directional
            # if (target, source) in edges:
            #     x0, y0, x1, y1 = _shift_edge(x0, y0, x1, y1, delta=-0.1*self.edge_artists[(source, target)].width)

            edge_paths[(source, target)] = np.c_[[x0, x1], [y0, y1]]

        return edge_paths


    def _update_selfloop_paths(self, edges):
        # restrict to self-loops
        edges = [(source, target) for source, target in edges if source == target]

        edge_paths = dict()
        for (source, target) in edges:
            edge_paths[(source, target)] = _get_selfloop_path(
                source,
                node_positions  = self.node_positions,
                selfloop_radius = self.edge_layout_kwargs['selfloop_radius'],
                origin          = self.edge_layout_kwargs['origin'],
                scale           = self.edge_layout_kwargs['scale']
            )
        return edge_paths


    def _update_curved_edge_paths(self, stale_edges):
        """Compute a new layout for curved edges keeping all other edges constant."""

        fixed_positions = dict()
        constant_edges = [edge for edge in self.edges if edge not in stale_edges]
        for edge in constant_edges:
            edge_artist = self.edge_artists[edge]
            if edge_artist.curved:
                for position in edge_artist.midline[1:-1]:
                    fixed_positions[uuid4()] = position
            else:
                # Densely sample points along the straight edge such that updated
                # edges avoid the whole edge, not just the end points.
                edge_origin = edge_artist.midline[0]
                delta = edge_artist.midline[-1] - edge_artist.midline[0]
                for ii in range(100):
                    # y = mx + b
                    m = (ii + 1) / (100 + 1)
                    fixed_positions[uuid4()] = m * delta + edge_origin
        fixed_positions.update(self.node_positions)

        return get_curved_edge_paths(stale_edges, fixed_positions, **self.edge_layout_kwargs)


    def _update_bundled_edge_paths(self, edges):
        # edge_paths = get_bundled_edge_paths(edges, self.node_positions, **self.edge_layout_kwargs)
        return get_bundled_edge_paths(self.edges, self.node_positions, **self.edge_layout_kwargs)


    def _initialize_node_label_offset(self, node_labels, node_label_offset):
        if isinstance(node_label_offset, (int, float)):
            node_label_offset = {node : node_label_offset * self._get_vector_pointing_outwards(self.node_positions[node]) for node in node_labels}
            recompute = True
            return node_label_offset, recompute
        elif isinstance(node_label_offset, (tuple, list, np.ndarray)):
            if len(node_label_offset) == 2:
                node_label_offset = {node : node_label_offset for node in node_labels}
                recompute = False
                return node_label_offset, recompute
            else:
                msg = "If the variable `node_label_offset` is an iterable, it should have length 2."
                msg+= f"Current length: {len(node_label_offset)}."
                raise ValueError(msg)
        else:
            msg = "The variable `node_label_offset` has to be either a float, an int, a tuple, a list, or a numpy ndarray."
            msg += f"\nCurrent type: {type(node_label_offset)}."
            raise TypeError(msg)


    def _get_centroid(self):
        return np.mean([position for position in self.node_positions.values()], axis=0)


    def _get_vector_pointing_outwards(self, xy):
        centroid = self._get_centroid()
        delta = xy - centroid
        distance = np.linalg.norm(delta)
        unit_vector = delta / distance
        return unit_vector


    def _initialize_node_label_fontdict(self, node_label_fontdict, node_labels, node_label_offset):
        if node_label_fontdict is None:
            node_label_fontdict = dict()

        node_label_fontdict.setdefault('horizontalalignment', 'center')
        node_label_fontdict.setdefault('verticalalignment', 'center')
        node_label_fontdict.setdefault('clip_on', False)
        node_label_fontdict.setdefault('zorder', np.inf)

        if np.all(np.isclose(node_label_offset, (0, 0))):
            # Labels are centered on node artists.
            # Set fontsize such that labels fit the diameter of the node artists.
            size = self._get_font_size(node_labels, node_label_fontdict) * 0.75 # conservative fudge factor
            node_label_fontdict.setdefault('size', size)

        return node_label_fontdict


    def _get_font_size(self, node_labels, node_label_fontdict):
        """
        Determine the maximum font size such that all labels fit inside their node artist.

        TODO:
        -----
        - potentially rescale font sizes individually on a per node basis
        """

        rescale_factor = np.inf
        for node, label in node_labels.items():
            artist = self.node_artists[node]
            diameter = 2 * (artist.radius - artist._lw_data/artist.linewidth_correction)
            width, height = _get_text_object_dimensions(self.ax, label, **node_label_fontdict)
            rescale_factor = min(rescale_factor, diameter/np.sqrt(width**2 + height**2))

        if 'size' in node_label_fontdict:
            size = rescale_factor * node_label_fontdict['size']
        else:
            size = rescale_factor * plt.rcParams['font.size']
        return size


    def draw_node_labels(self, node_labels, node_label_fontdict):
        """
        Draw node labels.

        Arguments
        ---------
        node_labels : dict key : str
           Mapping of nodes to labels.
           Only nodes in the dictionary are labelled.

        node_label_offset: 2-tuple or equivalent iterable (default (0., 0.))
            (x, y) offset from node centre of label position.

        node_label_fontdict : dict
            Keyword arguments passed to matplotlib.text.Text.
            For a full list of available arguments see the matplotlib documentation.
            The following default values differ from the defaults for matplotlib.text.Text:
                - size (adjusted to fit into node artists if offset is (0, 0))
                - horizontalalignment (default here: 'center')
                - verticalalignment (default here: 'center')
                - clip_on (default here: False)

        Updates:
        --------
        self.node_label_artists: dict
            Dictionary mapping nodes to text objects.

        """

        for node, label in node_labels.items():
            x, y = self.node_positions[node]
            dx, dy = self.node_label_offset[node]
            artist = self.ax.text(x+dx, y+dy, label, **node_label_fontdict)

            if node in self.node_label_artists:
                self.node_label_artists[node].remove()
            self.node_label_artists[node] = artist


    def _update_node_label_positions(self):
        if self._recompute_node_label_offsets:
            self._update_node_label_offsets()

        for node, (dx, dy) in self.node_label_offset.items():
            x, y = self.node_positions[node]
            self.node_label_artists[node].set_position((x + dx, y + dy))


    def _update_node_label_offsets(self, total_samples_per_edge=100):
        fixed = []
        for xy in self.node_positions.values():
            fixed.append(xy)
        for path in self.edge_paths.values():
            fixed.extend([_get_point_along_spline(path, fraction) for fraction in np.arange(0, 1, 1./total_samples_per_edge)])
        fixed = np.array(fixed)

        offsets = np.array(list(self.node_label_offset.values()))
        anchors = np.array([self.node_positions[node] for node in self.node_label_offset.keys()])

        offsets = self._optimise_offsets(anchors, offsets, fixed)

        for ii, node in enumerate(self.node_label_offset):
            self.node_label_offset[node] = offsets[ii]


    # # Variant no 1: use force directed layout to determine a suitable node label placements
    # # pros : labels repel each other
    # # cons : does not work very well; the optimum placement can still result in a collision
    # def _optimise_offsets(self, anchors, offsets, fixed, total_iterations=5):
    #     # Compute the net repulsion exerted on each label by nodes, edges and other labels.
    #     # Place the label in the direction of net repulsion at the desired distance from the corresponding node (anchor).
    #     # TODO Test if gradually stepping in the direction of net repulsion improves results.
    #     for ii in range(total_iterations):
    #         repulsion = self._get_repulsion(anchors + offsets, fixed)
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
    def _optimise_offsets(self, anchors, offsets, fixed, total_queries_per_point=360):
        tree = cKDTree(fixed)
        output = np.zeros_like(offsets)
        for ii, (anchor, offset) in enumerate(zip(anchors, offsets)):
            x = _get_n_points_on_a_circle(anchor, np.linalg.norm(offset), total_queries_per_point)
            # distances, _ = tree.query(x, 1) # can result in many ties; first element is arbitrarily chosen
            # output[ii] = x[np.argmax(distances)]
            distances, _ = tree.query(x, 2)
            output[ii] = x[np.argmax(np.sum(distances, axis=1))]
        return output - anchors


    def _initialize_edge_label_fontdict(self, edge_label_fontdict):
        if edge_label_fontdict is None:
            edge_label_fontdict = dict()

        edge_label_fontdict.setdefault('bbox', dict(boxstyle='round',
                                                    ec=(1.0, 1.0, 1.0),
                                                    fc=(1.0, 1.0, 1.0)))
        edge_label_fontdict.setdefault('horizontalalignment', 'center')
        edge_label_fontdict.setdefault('verticalalignment', 'center')
        edge_label_fontdict.setdefault('clip_on', False)
        edge_label_fontdict.setdefault('zorder', np.inf)
        return edge_label_fontdict


    def draw_edge_labels(self, edge_labels, edge_label_position,
                         edge_label_rotate, edge_label_fontdict):
        """
        Draw edge labels.

        Arguments
        ---------
        edge_labels : dict edge : str
            Mapping of edges to edge labels.
            Only edges in the dictionary are labelled.

        edge_label_position : float
            Relative position along the edge where the label is placed.
                head   : 0.
                centre : 0.5 (default)
                tail   : 1.

        edge_label_rotate : bool
            If True, edge labels are rotated such that they track the orientation of their edges.
            If False, edge labels are not rotated; the angle of the text is parallel to the axis.

        edge_label_fontdict : dict
            Keyword arguments passed to matplotlib.text.Text.

        Updates
        -------
        self.edge_label_artists: dict edge : text object
            Mapping of edges to edge label artists.

        """

        for edge, label in edge_labels.items():

            edge_artist = self.edge_artists[edge]

            if self._is_selfloop(edge) and (edge_artist.curved is False):
                msg = "Plotting of edge labels for self-loops not supported for straight edges."
                msg += "\nIgnoring edge with label: {}".format(label)
                warnings.warn(msg)
                continue

            x, y = _get_point_along_spline(edge_artist.midline, edge_label_position)

            if edge_label_rotate:

                # get tangent in degrees
                dx, dy = _get_tangent_at_point(edge_artist.midline, edge_label_position)
                angle = _get_angle(dx, dy, radians=True)

                # make label orientation "right-side-up"
                if angle > 90:
                    angle -= 180
                if angle < - 90:
                    angle += 180

            else:
                angle = None

            edge_label_artist = self.ax.text(x, y, label,
                                             rotation=angle,
                                             **edge_label_fontdict)

            if edge in self.edge_label_artists:
                self.edge_label_artists[edge].remove()
            self.edge_label_artists[edge] = edge_label_artist


    def _is_selfloop(self, edge):
        return True if edge[0] == edge[1] else False


    def _update_edge_label_positions(self, edges):

        labeled_edges = [edge for edge in edges if edge in self.edge_label_artists]

        for (n1, n2) in labeled_edges:

            edge_artist = self.edge_artists[(n1, n2)]

            if edge_artist.curved:
                x, y = _get_point_along_spline(edge_artist.midline, self.edge_label_position)
                dx, dy = _get_tangent_at_point(edge_artist.midline, self.edge_label_position)

            elif not edge_artist.curved and (n1 != n2):
                (x1, y1) = self.node_positions[n1]
                (x2, y2) = self.node_positions[n2]

                if (n2, n1) in self.edges: # i.e. bidirectional edge
                    x1, y1, x2, y2 = _shift_edge(x1, y1, x2, y2, delta=1.5*self.edge_artists[(n1, n2)].width)

                x, y = (x1 * self.edge_label_position + x2 * (1.0 - self.edge_label_position),
                        y1 * self.edge_label_position + y2 * (1.0 - self.edge_label_position))
                dx, dy = x2 - x1, y2 - y1

            else: # self-loop but edge is straight so we skip it
                pass

            self.edge_label_artists[(n1, n2)].set_position((x, y))

            if self.edge_label_rotate:
                angle = _get_angle(dx, dy, radians=True)
                # make label orientation "right-side-up"
                if angle > 90:
                    angle -= 180
                if angle < -90:
                    angle += 180
                # transform data coordinate angle to screen coordinate angle
                trans_angle = self.ax.transData.transform_angles(np.array((angle,)), np.atleast_2d((x, y)))[0]
                self.edge_label_artists[(n1, n2)].set_rotation(trans_angle)


    def _update_view(self):
        # Pad x and y limits as patches are not registered properly
        # when matplotlib sets axis limits automatically.
        # Hence we need to set them manually.

        # max_radius = np.max([artist.radius for artist in self.node_artists.values()])
        # maxx, maxy = np.max(list(self.node_positions.values()), axis=0)
        # minx, miny = np.min(list(self.node_positions.values()), axis=0)
        # w = maxx-minx
        # h = maxy-miny
        # padx, pady = 0.05*w + max_radius, 0.05*h + max_radius
        # corners = (minx-padx, miny-pady), (maxx+padx, maxy+pady)
        # self.ax.update_datalim(corners)

        self.ax.autoscale_view()
        self.ax.get_figure().canvas.draw()


class Graph(BaseGraph):
    """
    Parses the given graph and initialises the BaseGraph object.
    If the given graph includes edge weights, then these are mapped to
    colors using the `edge_cmap` parameter.

    Arguments:
    ----------
    graph: various formats
        Graph object to plot. Various input formats are supported.
        In order of precedence:
        - Edge list:
            Iterable of (source, target) or (source, target, weight) tuples,
            or equivalent (E, 2) or (E, 3) ndarray (where E is the number of edges).
        - Adjacency matrix:
            Full-rank (V, V) ndarray (where V is the number of nodes/vertices).
            The absence of a connection is indicated by a zero.
            Note that V > 3 as (2, 2) and (3, 3) matrices will be interpreted as edge lists.
        - networkx.Graph or igraph.Graph object

    node_layout : str or dict node : (float x, float y) (default 'spring')
        If node_layout is a string, the node positions are computed using
        the indicated method:
        - 'random'    : place nodes in random positions;
        - 'circular'  : place nodes regularly spaced on a circle;
        - 'spring'    : place nodes using a force-directed layout (Fruchterman-Reingold algorithm);
        - 'dot'       : place nodes using the Sugiyama algorithm; the graph should be directed and acyclic;
        - 'community' : place nodes such that nodes belonging to the same community are grouped together
        If node_layout is a dict, keys are nodes and values are (x, y) positions.

    node_layout_kwargs : dict (default None)
        Keyword arguments passed to node layout functions.
        See the documentation of the following functions for a full description
        of available options:
        - get_random_layout
        - get_circular_layout
        - get_fruchterman_reingold_layout
        - get_sugiyama_layout
        - get_community_layout

    node_shape : string or dict node : string (default 'o')
       The shape of the node. Specification is as for matplotlib.scatter
       marker, i.e. one of 'so^>v<dph8'.
       If a single string is provided all nodes will have the same shape.

    node_size : scalar or dict node : float (default 3.)
       Size (radius) of nodes.
       NOTE: Value is rescaled by BASE_SCALE (1e-2) to be compatible with layout routines in igraph and networkx.

    node_edge_width : scalar or dict node : float (default 0.5)
       Line width of node marker border.
       NOTE: Value is rescaled by BASE_SCALE (1e-2) to be compatible with layout routines in igraph and networkx.

    node_color : matplotlib color specification or dict node : color (default 'w')
       Node color.

    node_edge_color : matplotlib color specification or dict node : color (default DEFAULT_COLOR)
       Node edge color.

    node_alpha : scalar or dict node : float (default 1.)
       The node transparency.

    node_zorder : scalar or dict node : float (default 2)
       Order in which to plot the nodes.

    node_labels : bool or dict node : str (default False)
       If True, the nodes are labelled with their node IDs.
       If the node labels are supposed to be distinct from the node IDs,
       supply a dictionary mapping nodes to node labels.
       Only nodes in the dictionary are labelled.

    node_label_offset: scalar or 2-tuple or equivalent iterable (default (0., 0.))
        A (dx, dy) tuple specifies the exact offset from the node position.
        If a scalar delta is specified, the value is interpreted as a distance, and
        the label is placed delta away from the node position while trying to
        reduce node/label, node/edge, and label/label overlaps.

    node_label_fontdict : dict
        Keyword arguments passed to matplotlib.text.Text.
        For a full list of available arguments see the matplotlib documentation.
        The following default values differ from the defaults for matplotlib.text.Text:
            - size (adjusted to fit into node artists if offset is (0, 0))
            - horizontalalignment (default here: 'center')
            - verticalalignment (default here: 'center')
            - clip_on (default here: False)
            - zorder (default here: inf)

    edge_width : float or dict (source, target) : width (default 1.)
        Line width of edges.
        NOTE: Value is rescaled by BASE_SCALE (1e-2) to be compatible with layout routines in igraph and networkx.

    edge_cmap : matplotlib color map (default RdGy)
        Color map used to map edge weights to edge colors. Should be diverging.
        If edge weights are strictly positive, weights are mapped to the
        left hand side of the color map with vmin=0 and vmax=np.max(weights).
        If edge weights are positive and negative, then weights are mapped
        to colors such that a weight of zero corresponds to the center of the
        color map; the boundaries are set to +/- the maximum absolute weight.
        If the graph is unweighted or the edge colors are specified explicitly,
        this parameter is ignored.

    edge_color : matplotlib color specification or dict (source, target) : color (default DEFAULT_COLOR)
       Edge color. Takes precent over edge_cmap.

    edge_alpha : float or dict (source, target) : float (default 1.)
        The edge transparency,

    edge_zorder : int or dict (source, target) : int (default 1)
        Order in which to plot the edges.
        If None, the edges will be plotted in the order they appear in 'adjacency'.
        Note: graphs typically appear more visually pleasing if darker edges
        are plotted on top of lighter edges.

    arrows : bool, optional (default False)
        If True, draw edges with arrow heads.

    edge_layout : str or dict edge : segments (default 'straight')
        If edge_layout is a string, determine the layout internally:
        - 'straight' : draw edges as straight lines
        - 'curved'   : draw edges as curved splines; the spline control points are optimised to avoid other nodes and edges
        - 'bundled'  : draw edges as edge bundles
        If edge_layout is a dict, the keys are edges and the
        values are edge paths in the form iterables of (x, y)
        tuples, the edge segments.

    edge_layout_kwargs : dict (default None)
        Keyword arguments passed to edge layout functions.
        See the documentation of the following functions for a full description
        of available options:
        - get_straight_edge_paths
        - get_curved_edge_paths
        - get_bundled_edge_paths

    edge_labels : bool or dict edge : str
        If True, the edges are labelled with their edge IDs.
        If the edge labels are supposed to be distinct from the edge IDs,
        supply a dictionary mapping edges to edge labels.
        Only edges in the dictionary are labelled.

    edge_label_position : float
        Relative position along the edge where the label is placed.
            head   : 0.
            centre : 0.5 (default)
            tail   : 1.

    edge_label_rotate : bool (default True)
        If True, edge labels are rotated such that they track the orientation of their edges.
        If False, edge labels are not rotated; the angle of the text is parallel to the axis.

    edge_label_fontdict : dict
        Keyword arguments passed to matplotlib.text.Text.
        For a full list of available arguments see the matplotlib documentation.
        The following default values differ from the defaults for matplotlib.text.Text:
            - horizontalalignment (default here: 'center'),
            - verticalalignment (default here: 'center')
            - clip_on (default here: False),
            - bbox (default here: dict(boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)),
            - zorder (default here: inf),
            - rotation (determined by edge_label_rotate argument)

    origin : (float x, float y) tuple or None (default (0, 0))
        The lower left hand corner of the bounding box specifying the extent of the canvas.

    scale : (float delta x, float delta y) or None (default (1, 1))
        The width and height of the bounding box specifying the extent of the canvas.

    prettify : bool (default True)
        If True, despine and remove ticks and tick labels.
        Set figure background to white. Set axis aspect to equal.

    ax : matplotlib.axis instance or None (default None)
       Axis to plot onto; if none specified, one will be instantiated with plt.gca().


    Attributes:
    -----------
    node_artists : dict node : NodeArtist instance
        The node artists.

    edge_artists : dict edge : EdgeArtist instance
        The edge artists.

    node_label_artists : dict node : matplotlib.Text
        The node label text objects (if applicable).

    edge_label_artists : dict edge : matplotlib.Text
        The edge label text objects (if applicable).

    node_positions : dict node : (x, y) tuple
        The node positions.

    """

    def __init__(self, graph, edge_cmap='RdGy', *args, **kwargs):

        # Accept a variety of formats for 'graph' and convert to common denominator.
        nodes, edges, edge_weight = parse_graph(graph)
        kwargs.setdefault('nodes', nodes)

        # Color and reorder edges for weighted graphs.
        if edge_weight:
            # If the graph is weighted, we want to visualise the weights using color.
            # Edge width is another popular choice when visualising weighted networks,
            # but if the variance in weights is large, this typically results in less
            # visually pleasing results.
            edge_color = _get_color(edge_weight, cmap=edge_cmap)

            # Plotting darker edges over lighter edges typically results in visually
            # more pleasing results. Here we hence specify the relative order in
            # which edges are plotted according to the color of the edge.
            edge_zorder = _get_zorder(edge_color)
            node_zorder = np.max(list(edge_zorder.values())) + 1

            kwargs.setdefault('edge_color', edge_color)
            kwargs.setdefault('edge_zorder', edge_zorder)
            kwargs.setdefault('node_zorder', node_zorder)

        super().__init__(edges, *args, **kwargs)


class ClickableArtists(object):
    """Implements selection of matplotlib artists via the mouse left click (+/- ctrl or command key).

    Notes:
    ------
    Methods adapted with some modifications from:
    https://stackoverflow.com/questions/47293499/window-select-multiple-artists-and-drag-them-on-canvas/47312637#47312637
    """
    def __init__(self, artists):

        try:
            self.fig, = set(list(artist.figure for artist in artists))
        except ValueError:
            raise Exception("All artists have to be on the same figure!")

        try:
            self.ax, = set(list(artist.axes for artist in artists))
        except ValueError:
            raise Exception("All artists have to be on the same axis!")

        # self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)

        self._clickable_artists = list(artists)
        self._selected_artists = []
        self._base_linewidth = dict([(artist, artist._lw_data) for artist in artists])
        self._base_edgecolor = dict([(artist, artist.get_edgecolor()) for artist in artists])

        if matplotlib.get_backend() == 'MacOSX':
            msg  = "You appear to be using the MacOSX backend."
            msg += "\nModifier key presses are bugged on this backend. See https://github.com/matplotlib/matplotlib/issues/20486"
            msg += "\nConsider using a different backend, e.g. TkAgg (import matplotlib; matplotlib.use('TkAgg'))."
            msg += "\nNote that you must set the backend before importing any package depending on matplotlib (includes pyplot, networkx, netgraph)."
            warnings.warn(msg)

    # def _on_press(self, event):
    def _on_release(self, event):
        if event.inaxes == self.ax:
            for artist in self._clickable_artists:
                if artist.contains(event)[0]:
                    if event.key in ('control', 'super+??', 'ctrl+??'):
                        self._toggle_select_artist(artist)
                    else:
                        self._deselect_all_other_artists(artist)
                        self._toggle_select_artist(artist)
                        # NOTE: if two artists are overlapping, only the first one encountered is selected!
                    break
            else:
                if not event.key in ('control', 'super+??', 'ctrl+??'):
                    self._deselect_all_artists()
        else:
            print("Warning: clicked outside axis limits!")


    def _toggle_select_artist(self, artist):
        if artist in self._selected_artists:
            self._deselect_artist(artist)
        else:
            self._select_artist(artist)


    def _select_artist(self, artist):
        if not (artist in self._selected_artists):
            linewidth = artist._lw_data
            artist.set_linewidth(max(1.5 * linewidth, 0.003))
            artist.set_edgecolor('black')
            self._selected_artists.append(artist)
            self.fig.canvas.draw_idle()


    def _deselect_artist(self, artist):
        if artist in self._selected_artists: # should always be true?
            artist.set_linewidth(self._base_linewidth[artist])
            artist.set_edgecolor(self._base_edgecolor[artist])
            self._selected_artists.remove(artist)
            self.fig.canvas.draw_idle()


    def _deselect_all_artists(self):
        for artist in self._selected_artists[:]: # we make a copy of the list with [:], as we are modifying the list being iterated over
            self._deselect_artist(artist)


    def _deselect_all_other_artists(self, artist_to_keep):
        for artist in self._selected_artists[:]:
            if artist != artist_to_keep:
                self._deselect_artist(artist)


class SelectableArtists(ClickableArtists):
    """Augments ClickableArtists with a rectangle selector.

    Notes:
    ------
    Methods adapted with some modifications from:
    https://stackoverflow.com/questions/47293499/window-select-multiple-artists-and-drag-them-on-canvas/47312637#47312637
    """
    def __init__(self, artists):
        super().__init__(artists)

        self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        # self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('motion_notify_event',  self._on_motion)

        self._selectable_artists = list(artists)
        self._currently_selecting = False

        self._rect = plt.Rectangle((0, 0), 1, 1, linestyle="--", edgecolor="crimson", fill=False)
        self.ax.add_patch(self._rect)
        self._rect.set_visible(False)

        self._x0 = 0
        self._y0 = 0
        self._x1 = 0
        self._y1 = 0


    def _on_press(self, event):
        # super()._on_press(event)

        if event.inaxes == self.ax:
            # reset rectangle
            self._x0 = event.xdata
            self._y0 = event.ydata
            self._x1 = event.xdata
            self._y1 = event.ydata

            for artist in self._clickable_artists:
                if artist.contains(event)[0]:
                    break
            else:
                self._currently_selecting = True


    def _on_release(self, event):
        super()._on_release(event)

        if self._currently_selecting:
            # select artists inside window
            for artist in self._selectable_artists:
                if isinstance(artist, NodeArtist):
                    if self._is_inside_rect(*artist.xy):
                        if event.key in ('control', 'super+??', 'ctrl+??'): # if/else probably superfluouos
                            self._toggle_select_artist(artist)              # as no artists will be selected
                        else:                                               # if control is not held previously
                            self._select_artist(artist)                     #
                elif isinstance(artist, EdgeArtist):
                    if np.all([self._is_inside_rect(x, y) for x, y in artist.midline]):
                        if event.key in ('control', 'super+??', 'ctrl+??'): # if/else probably superfluouos
                            self._toggle_select_artist(artist)              # as no artists will be selected
                        else:                                               # if control is not held previously
                            self._select_artist(artist)                     #

            # stop window selection and draw new state
            self._currently_selecting = False
            self._rect.set_visible(False)
            self.fig.canvas.draw_idle()


    def _on_motion(self, event):
        if event.inaxes == self.ax:
            if self._currently_selecting:
                self._x1 = event.xdata
                self._y1 = event.ydata
                # add rectangle for selection here
                self._selector_on()


    def _is_inside_rect(self, x, y):
        xlim = np.sort([self._x0, self._x1])
        ylim = np.sort([self._y0, self._y1])
        if (xlim[0]<=x) and (x<xlim[1]) and (ylim[0]<=y) and (y<ylim[1]):
            return True
        else:
            return False


    def _selector_on(self):
        self._rect.set_visible(True)
        xlim = np.sort([self._x0, self._x1])
        ylim = np.sort([self._y0, self._y1])
        self._rect.set_xy((xlim[0],ylim[0] ) )
        self._rect.set_width(np.diff(xlim))
        self._rect.set_height(np.diff(ylim))
        self.fig.canvas.draw_idle()


class DraggableArtists(SelectableArtists):
    """Augments SelectableArtists to support dragging of artists by holding the left mouse button.

    Notes:
    ------
    Methods adapted with some modifications from:
    https://stackoverflow.com/questions/47293499/window-select-multiple-artists-and-drag-them-on-canvas/47312637#47312637
    """

    def __init__(self, artists):
        super().__init__(artists)

        self._draggable_artists = list(artists)
        self._currently_clicking_on_artist = None
        self._currently_dragging = False
        self._offset = dict()


    def _on_press(self, event):
        super()._on_press(event)

        if event.inaxes == self.ax:
            for artist in self._draggable_artists:
                if artist.contains(event)[0]:
                    self._currently_clicking_on_artist = artist
                    break
        else:
            print("Warning: clicked outside axis limits!")


    def _on_motion(self, event):
        super()._on_motion(event)

        if event.inaxes == self.ax:
            if self._currently_clicking_on_artist:
                if self._currently_clicking_on_artist not in self._selected_artists:
                    if event.key not in ('control', 'super+??', 'ctrl+??'):
                        self._deselect_all_artists()
                    self._select_artist(self._currently_clicking_on_artist)
                self._offset = {artist : artist.xy - np.array([event.xdata, event.ydata]) for artist in self._selected_artists if artist in self._draggable_artists}
                self._currently_clicking_on_artist = None
                self._currently_dragging = True

            if self._currently_dragging:
                self._move(event)


    def _on_release(self, event):
        if self._currently_dragging:
            self._currently_dragging = False
        else:
            self._currently_clicking_on_artist = None
            super()._on_release(event)


    def _move(self, event):
        cursor_position = np.array([event.xdata, event.ydata])
        for artist in self._selected_artists:
            artist.xy = cursor_position + self._offset[artist]
        self.fig.canvas.draw_idle()


class DraggableGraph(Graph, DraggableArtists):

    def __init__(self, *args, **kwargs):
        Graph.__init__(self, *args, **kwargs)
        DraggableArtists.__init__(self, self.node_artists.values())

        self._draggable_artist_to_node = dict(zip(self.node_artists.values(), self.node_artists.keys()))

        self._clickable_artists.extend(list(self.edge_artists.values()))
        self._selectable_artists.extend(list(self.edge_artists.values()))
        self._base_linewidth.update(dict([(artist, artist._lw_data) for artist in self.edge_artists.values()]))
        self._base_edgecolor.update(dict([(artist, artist.get_edgecolor()) for artist in self.edge_artists.values()]))

        # # trigger resize of labels when canvas size changes
        # self.fig.canvas.mpl_connect('resize_event', self._on_resize)


    def _move(self, event):

        cursor_position = np.array([event.xdata, event.ydata])

        nodes = self._get_stale_nodes()
        self._update_node_positions(nodes, cursor_position)
        self._update_node_artists(nodes)

        if hasattr(self, 'node_label_artists'):
            self._update_node_label_positions()

        edges = self._get_stale_edges(nodes)
        # In the interest of speed, we only compute the straight edge paths here.
        # We will re-compute other edge layouts only on mouse button release,
        # i.e. when the dragging motion has stopped.
        edge_paths = dict()
        edge_paths.update(self._update_straight_edge_paths([(source, target) for (source, target) in edges if source != target]))
        edge_paths.update(self._update_selfloop_paths([(source, target) for (source, target) in edges if source == target]))
        self.edge_paths.update(edge_paths)
        self._update_edge_artists(edge_paths)

        if hasattr(self, 'edge_label_artists'):
            self._update_edge_label_positions(edges)

        self.fig.canvas.draw_idle()


    def _get_stale_nodes(self):
        return [self._draggable_artist_to_node[artist] for artist in self._selected_artists if artist in self._draggable_artists]


    def _update_node_positions(self, nodes, cursor_position):
        for node in nodes:
            self.node_positions[node] = cursor_position + self._offset[self.node_artists[node]]


    def _get_stale_edges(self, nodes=None):
        if nodes is None:
            nodes = self._get_stale_nodes()
        return [(source, target) for (source, target) in self.edges if (source in nodes) or (target in nodes)]


    def _on_release(self, event):
        if self._currently_dragging and not (self.edge_layout == 'straight'):
            nodes = self._get_stale_nodes()
            edges = self._get_stale_edges(nodes)
            self._update_edges(edges)

            if hasattr(self, 'edge_label_artists'): # move edge labels
                self._update_edge_label_positions(edges)

        super()._on_release(event)


    # def _on_resize(self, event):
    #     if hasattr(self, 'node_labels'):
    #         self.draw_node_labels(self.node_labels)
    #         # print("As node label font size was not explicitly set, automatically adjusted node label font size to {:.2f}.".format(self.node_label_font_size))


class EmphasizeOnHover(object):

    def __init__(self, artists):

        self.emphasizeable_artists = artists
        self._base_alpha = {artist : artist.get_alpha() for artist in self.emphasizeable_artists}
        self.deemphasized_artists = []

        try:
            self.fig, = set(list(artist.figure for artist in artists))
        except ValueError:
            raise Exception("All artists have to be on the same figure!")

        try:
            self.ax, = set(list(artist.axes for artist in artists))
        except ValueError:
            raise Exception("All artists have to be on the same axis!")

        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)


    def _on_motion(self, event):

        if event.inaxes == self.ax:
            # on artist
            selected_artist = None
            for artist in self.emphasizeable_artists:
                if artist.contains(event)[0]: # returns two arguments for some reason
                    selected_artist = artist
                    break

            if selected_artist:
                for artist in self.emphasizeable_artists:
                    if artist is not selected_artist:
                        artist.set_alpha(self._base_alpha[artist]/5)
                        self.deemphasized_artists.append(artist)
                self.fig.canvas.draw_idle()

            # not on any artist
            if (selected_artist is None) and self.deemphasized_artists:
                for artist in self.deemphasized_artists:
                    artist.set_alpha(self._base_alpha[artist])
                self.deemphasized_artists = []
                self.fig.canvas.draw_idle()


class EmphasizeOnHoverGraph(Graph, EmphasizeOnHover):

    def __init__(self, *args, **kwargs):
        Graph.__init__(self, *args, **kwargs)

        artists = list(self.node_artists.values()) + list(self.edge_artists.values())
        keys = list(self.node_artists.keys()) + list(self.edge_artists.keys())
        self.artist_to_key = dict(zip(artists, keys))
        EmphasizeOnHover.__init__(self, artists)


    def _on_motion(self, event):

        if event.inaxes == self.ax:

            # determine if the cursor is on an artist
            selected_artist = None
            for artist in self.emphasizeable_artists:
                if artist.contains(event)[0]: # returns bool, {} for some reason
                    selected_artist = artist
                    break

            if selected_artist:
                emphasized_artists = [selected_artist]

                if isinstance(selected_artist, NodeArtist):
                    node = self.artist_to_key[selected_artist]
                    connected_edges = [edge for edge in self.edge_artists.keys() if node in edge]
                    edge_artists = [self.edge_artists[edge] for edge in connected_edges]
                    emphasized_artists.extend(edge_artists)
                    neighbours = _get_unique_nodes(connected_edges)
                    node_artists = [self.node_artists[neighbour] for neighbour in neighbours if neighbour != node]
                    emphasized_artists.extend(node_artists)

                elif isinstance(selected_artist, EdgeArtist):
                    source, target = self.artist_to_key[selected_artist]
                    emphasized_artists.append(self.node_artists[source])
                    emphasized_artists.append(self.node_artists[target])

                for artist in self.emphasizeable_artists:
                    if artist not in emphasized_artists:
                        artist.set_alpha(self._base_alpha[artist]/5)
                        self.deemphasized_artists.append(artist)
                self.fig.canvas.draw_idle()

            # not on any artist
            if (selected_artist is None) and self.deemphasized_artists:
                for artist in self.deemphasized_artists:
                    artist.set_alpha(self._base_alpha[artist])
                self.deemphasized_artists = []
                self.fig.canvas.draw_idle()


class AnnotateOnClick(object):

    def __init__(self, artist_to_annotation, annotation_fontdict=None):

        self.artist_to_annotation = artist_to_annotation
        self.annotated_artists = set()
        self.artist_to_text_object = dict()
        self.annotation_fontdict = dict(
            backgroundcolor = 'white',
            zorder          = np.inf,
            clip_on         = False
        )
        if annotation_fontdict:
            self.annotation_fontdict.update(annotation_fontdict)

        self.fig.canvas.mpl_connect("button_release_event", self._on_release)


    def _on_release(self, event):

        if event.inaxes == self.ax:

            # clicked on already annotated artist
            for artist in self.annotated_artists:
                if artist.contains(event)[0]:
                    self._remove_annotation(artist)
                    self.fig.canvas.draw()
                    return

            # clicked on un-annotated artist
            for artist in self.artist_to_annotation:
                if artist.contains(event)[0]:
                    placement = self._get_annotation_placement(artist)
                    self._add_annotation(artist, *placement)
                    self.fig.canvas.draw()
                    return

            # # clicked outside of any artist
            # for artist in list(self.annotated_artists): # list to force copy
            #     self._remove_annotation(artist)
            # self.fig.canvas.draw()


    def _get_annotation_placement(self, artist):
        vector = self._get_vector_pointing_outwards(artist.xy)
        x, y = artist.xy + 2 * artist.radius * vector
        horizontalalignment, verticalalignment = self._get_text_alignment(vector)
        return x, y, horizontalalignment, verticalalignment


    def _get_centroid(self):
        return np.mean([artist.xy for artist in self.artist_to_annotation], axis=0)


    def _get_vector_pointing_outwards(self, xy):
        centroid = self._get_centroid()
        delta = xy - centroid
        distance = np.linalg.norm(delta)
        unit_vector = delta / distance
        return unit_vector


    def _get_text_alignment(self, vector):
        dx, dy = vector
        angle = _get_angle(dx, dy, radians=True) % 360

        if (45 <= angle < 135):
            horizontalalignment = 'center'
            verticalalignment = 'bottom'
        elif (135 <= angle < 225):
            horizontalalignment = 'right'
            verticalalignment = 'center'
        elif (225 <= angle < 315):
            horizontalalignment = 'center'
            verticalalignment = 'top'
        else:
            horizontalalignment = 'left'
            verticalalignment = 'center'

        return horizontalalignment, verticalalignment


    def _add_annotation(self, artist, x, y, horizontalalignment, verticalalignment):
        params = self.annotation_fontdict.copy()
        params.setdefault('horizontalalignment', horizontalalignment)
        params.setdefault('verticalalignment', verticalalignment)
        if isinstance(self.artist_to_annotation[artist], str):
            self.artist_to_text_object[artist] = self.ax.text(
                x, y, self.artist_to_annotation[artist], **params)
        elif isinstance(self.artist_to_annotation[artist], dict):
            params.update(self.artist_to_annotation[artist].copy())
            self.artist_to_text_object[artist] = self.ax.text(
                x, y, **params
            )
        self.annotated_artists.add(artist)


    def _remove_annotation(self, artist):
        text_object = self.artist_to_text_object[artist]
        text_object.remove()
        del self.artist_to_text_object[artist]
        self.annotated_artists.discard(artist)


class AnnotateOnClickGraph(Graph, AnnotateOnClick):

    def __init__(self, *args, **kwargs):
        Graph.__init__(self, *args, **kwargs)

        artist_to_annotation = dict()
        if 'annotations' in kwargs:
            for key, annotation in kwargs['annotations'].items():
                if key in self.nodes:
                    artist_to_annotation[self.node_artists[key]] = annotation
                elif key in self.edges:
                    artist_to_annotation[self.edge_artists[key]] = annotation
                else:
                    raise ValueError(f"There is no node or edge with the ID {key} for the annotation '{annotation}'.")

        AnnotateOnClick.__init__(self, artist_to_annotation)


    def _get_centroid(self):
        return Graph._get_centroid(self)


    def _get_annotation_placement(self, artist):
        if isinstance(artist, NodeArtist):
            return self._get_node_annotation_placement(artist)
        elif isinstance(artist, EdgeArtist):
            return self._get_edge_annotation_placement(artist)
        else:
            raise NotImplementedError


    def _get_node_annotation_placement(self, artist):
        return super()._get_annotation_placement(artist)


    def _get_edge_annotation_placement(self, artist):
        midpoint = _get_point_along_spline(artist.midline, 0.5)

        tangent = _get_tangent_at_point(artist.midline, 0.5)
        orthogonal_vector = _get_orthogonal_unit_vector(np.atleast_2d(tangent)).ravel()
        vector_pointing_outwards = self._get_vector_pointing_outwards(midpoint)
        if _get_interior_angle_between(orthogonal_vector, vector_pointing_outwards, radians=True) > 90:
            orthogonal_vector *= -1

        x, y = midpoint + 2 * artist.width * orthogonal_vector
        horizontalalignment, verticalalignment = self._get_text_alignment(orthogonal_vector)
        return x, y, horizontalalignment, verticalalignment


class TableOnClick(object):

    def __init__(self, artist_to_table, table_kwargs=None):

        self.artist_to_table = artist_to_table
        self.table = None
        self.table_fontsize = None
        self.table_kwargs = dict(
            # bbox = [1.1, 0.1, 0.5, 0.8],
            # edges = 'horizontal',
        )

        if table_kwargs:
            if 'fontsize' in table_kwargs:
                self.table_fontsize = table_kwargs['fontsize']
            self.table_kwargs.update(table_kwargs)

        try:
            self.fig, = set(list(artist.figure for artist in artist_to_table))
        except ValueError:
            raise Exception("All artists have to be on the same figure!")

        try:
            self.ax, = set(list(artist.axes for artist in artist_to_table))
        except ValueError:
            raise Exception("All artists have to be on the same axis!")

        self.fig.canvas.mpl_connect("button_release_event", self._on_release)


    def _on_release(self, event):
        if event.inaxes == self.ax:
            for artist in self.artist_to_table:
                if artist.contains(event)[0]:
                    if self.table:
                        self._remove_table()
                    self._add_table(artist)
                    self.fig.canvas.draw()
                    break
            else:
                if self.table:
                    self._remove_table()
                    self.fig.canvas.draw()


    def _add_table(self, artist):
        df = self.artist_to_table[artist]
        self.table = self.ax.table(
            cellText = df.values.tolist(),
            rowLabels = df.index.values,
            colLabels = df.columns.values,
            **self.table_kwargs,
        )

        if self.table_fontsize:
            self.table.auto_set_font_size(False)
            self.table.set_fontsize(self.table_fontsize)

    def _remove_table(self):
        self.table.remove()
        self.table = None


class TableOnClickGraph(Graph, TableOnClick):

    def __init__(self, *args, **kwargs):
        Graph.__init__(self, *args, **kwargs)

        artist_to_table = dict()
        if 'tables' in kwargs:
            for key, table in kwargs['tables'].items():
                if key in self.nodes:
                    artist_to_table[self.node_artists[key]] = table
                elif key in self.edges:
                    artist_to_table[self.edge_artists[key]] = table
                else:
                    raise ValueError(f"There is no node or edge with the ID {key} for the table '{table}'.")

        if 'table_kwargs' in kwargs:
            TableOnClick.__init__(self, artist_to_table, kwargs['table_kwargs'])
        else:
            TableOnClick.__init__(self, artist_to_table)


class InteractiveGraph(DraggableGraph, EmphasizeOnHoverGraph, AnnotateOnClickGraph, TableOnClickGraph):
    """
    Initializes an interactive Graph instance.

    Arguments:
    ----------
    graph: various formats
        Graph object to plot. Various input formats are supported.
        In order of precedence:
        - Edge list:
            Iterable of (source, target) or (source, target, weight) tuples,
            or equivalent (E, 2) or (E, 3) ndarray (where E is the number of edges).
        - Adjacency matrix:
            Full-rank (V, V) ndarray (where V is the number of nodes/vertices).
            The absence of a connection is indicated by a zero.
            Note that V > 3 as (2, 2) and (3, 3) matrices will be interpreted as edge lists.
        - networkx.Graph or igraph.Graph object

    node_layout : str or dict node : (float x, float y) (default 'spring')
        If node_layout is a string, the node positions are computed using
        the indicated method:
        - 'random'    : place nodes in random positions;
        - 'circular'  : place nodes regularly spaced on a circle;
        - 'spring'    : place nodes using a force-directed layout (Fruchterman-Reingold algorithm);
        - 'dot'       : place nodes using the Sugiyama algorithm; the graph should be directed and acyclic;
        - 'community' : place nodes such that nodes belonging to the same community are grouped together
        If node_layout is a dict, keys are nodes and values are (x, y) positions.

    node_layout_kwargs : dict (default None)
        Keyword arguments passed to node layout functions.
        See the documentation of the following functions for a full description
        of available options:
        - get_random_layout
        - get_circular_layout
        - get_fruchterman_reingold_layout
        - get_sugiyama_layout
        - get_community_layout

    node_shape : string or dict node : string (default 'o')
       The shape of the node. Specification is as for matplotlib.scatter
       marker, i.e. one of 'so^>v<dph8'.
       If a single string is provided all nodes will have the same shape.

    node_size : scalar or dict node : float (default 3.)
       Size (radius) of nodes.
       NOTE: Value is rescaled by BASE_SCALE (1e-2) to be compatible with layout routines in igraph and networkx.

    node_edge_width : scalar or dict node : float (default 0.5)
       Line width of node marker border.
       NOTE: Value is rescaled by BASE_SCALE (1e-2) to be compatible with layout routines in igraph and networkx.

    node_color : matplotlib color specification or dict node : color (default 'w')
       Node color.

    node_edge_color : matplotlib color specification or dict node : color (default DEFAULT_COLOR)
       Node edge color.

    node_alpha : scalar or dict node : float (default 1.)
       The node transparency.

    node_zorder : scalar or dict node : float (default 2)
       Order in which to plot the nodes.

    node_labels : bool or dict node : str (default False)
       If True, the nodes are labelled with their node IDs.
       If the node labels are supposed to be distinct from the node IDs,
       supply a dictionary mapping nodes to node labels.
       Only nodes in the dictionary are labelled.

    node_label_offset: scalar or 2-tuple or equivalent iterable (default (0., 0.))
        A (dx, dy) tuple specifies the exact offset from the node position.
        If a scalar delta is specified, the value is interpreted as a distance, and
        the label is placed delta away from the node position while trying to
        reduce node/label, node/edge, and label/label overlaps.

    node_label_fontdict : dict
        Keyword arguments passed to matplotlib.text.Text.
        For a full list of available arguments see the matplotlib documentation.
        The following default values differ from the defaults for matplotlib.text.Text:
            - size (adjusted to fit into node artists if offset is (0, 0))
            - horizontalalignment (default here: 'center')
            - verticalalignment (default here: 'center')
            - clip_on (default here: False)
            - zorder (default here: inf)

    edge_width : float or dict (source, target) : width (default 1.)
        Line width of edges.
        NOTE: Value is rescaled by BASE_SCALE (1e-2) to be compatible with layout routines in igraph and networkx.

    edge_cmap : matplotlib color map (default RdGy)
        Color map used to map edge weights to edge colors. Should be diverging.
        If edge weights are strictly positive, weights are mapped to the
        left hand side of the color map with vmin=0 and vmax=np.max(weights).
        If edge weights are positive and negative, then weights are mapped
        to colors such that a weight of zero corresponds to the center of the
        color map; the boundaries are set to +/- the maximum absolute weight.
        If the graph is unweighted or the edge colors are specified explicitly,
        this parameter is ignored.

    edge_color : matplotlib color specification or dict (source, target) : color (default DEFAULT_COLOR)
       Edge color.

    edge_alpha : float or dict (source, target) : float (default 1.)
        The edge transparency,

    edge_zorder : int or dict (source, target) : int (default 1)
        Order in which to plot the edges.
        If None, the edges will be plotted in the order they appear in 'adjacency'.
        Note: graphs typically appear more visually pleasing if darker edges
        are plotted on top of lighter edges.

    arrows : bool, optional (default False)
        If True, draw edges with arrow heads.

    edge_layout : str or dict edge : segments (default 'straight')
        If edge_layout is a string, determine the layout internally:
        - 'straight' : draw edges as straight lines
        - 'curved'   : draw edges as curved splines; the spline control points are optimised to avoid other nodes and edges
        - 'bundled'  : draw edges as edge bundles
        If edge_layout is a dict, the keys are edges and the
        values are edge paths in the form iterables of (x, y)
        tuples, the edge segments.

    edge_layout_kwargs : dict (default None)
        Keyword arguments passed to edge layout functions.
        See the documentation of the following functions for a full description
        of available options:
        - get_straight_edge_paths
        - get_curved_edge_paths
        - get_bundled_edge_paths

    edge_labels : bool or dict edge : str
        If True, the edges are labelled with their edge IDs.
        If the edge labels are supposed to be distinct from the edge IDs,
        supply a dictionary mapping edges to edge labels.
        Only edges in the dictionary are labelled.

    edge_label_position : float
        Relative position along the edge where the label is placed.
            head   : 0.
            centre : 0.5 (default)
            tail   : 1.

    edge_label_rotate : bool (default True)
        If True, edge labels are rotated such that they track the orientation of their edges.
        If False, edge labels are not rotated; the angle of the text is parallel to the axis.

    edge_label_fontdict : dict
        Keyword arguments passed to matplotlib.text.Text.
        For a full list of available arguments see the matplotlib documentation.
        The following default values differ from the defaults for matplotlib.text.Text:
            - horizontalalignment (default here: 'center'),
            - verticalalignment (default here: 'center')
            - clip_on (default here: False),
            - bbox (default here: dict(boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)),
            - zorder (default here: inf),
            - rotation (determined by edge_label_rotate argument)

    annotations : dict node/edge : string or dict
        Additional information that can be revealed/hidden by clicking on the corresponding node or edge.
        annotations = {
            0      : 'Normal node',
            1      : {s : 'Less important node', fontsize : 2},
            2      : {s : 'Very important node', fontcolor : 'red'},
            (0, 1) : 'Normal edge',
            (1, 2) : {s : 'Less important edge', fontsize : 2},
            (2, 0) : {s : 'Very important edge', fontcolor : 'red'},
        }

    annotation_fontdict : dict
        Keyword arguments passed to matplotlib.text.Text if only the annotation string is given.
        For a full list of available arguments see the matplotlib documentation.
        The following default values differ from the defaults for matplotlib.text.Text:
            - horizontalalignment (depends on node position or edge orientation),
            - verticalalignment (depends on node position or edge orientation),
            - clip_on (default here: False),
            - backgroundcolor (default here: 'white'),
            - zorder (default here: inf),

    tables : dict node/edge : pandas dataframe
        Tabular information that can be displayed by clicking on the corresponding node or edge.

    table_kwargs : dict
        Keyword arguments passed to matplotlib.pyplot.table.

    origin : (float x, float y) tuple or None (default (0, 0))
        The lower left hand corner of the bounding box specifying the extent of the canvas.

    scale : (float delta x, float delta y) or None (default (1, 1))
        The width and height of the bounding box specifying the extent of the canvas.

    prettify : bool (default True)
        If True, despine and remove ticks and tick labels.
        Set figure background to white. Set axis aspect to equal.

    ax : matplotlib.axis instance or None (default None)
       Axis to plot onto; if none specified, one will be instantiated with plt.gca().


    Attributes:
    -----------
    node_artists : dict node : NodeArtist instance
        The node artists.

    edge_artists : dict edge : EdgeArtist instance
        The edge artists.

    node_label_artists : dict node : matplotlib.Text
        The node label text objects (if applicable).

    edge_label_artists : dict edge : matplotlib.Text
        The edge label text objects (if applicable).

    node_positions : dict node : (x, y) tuple
        The node positions.


    NOTE:
    -----
    You must retain a reference to the plot instance!
    Otherwise, the plot instance will be garbage collected after the initial draw
    and you won't be able to move the plot elements around.


    Example:
    --------
    >>> import matplotlib.pyplot as plt
    >>> from netgraph import InteractiveGraph
    >>> plt.ion()
    >>> plot_instance = InteractiveGraph(graph_obj)
    >>> plt.show()

    """

    def __init__(self, *args, **kwargs):

        DraggableGraph.__init__(self, *args, **kwargs)

        artists = list(self.node_artists.values()) + list(self.edge_artists.values())
        keys = list(self.node_artists.keys()) + list(self.edge_artists.keys())
        self.artist_to_key = dict(zip(artists, keys))
        EmphasizeOnHover.__init__(self, artists)

        artist_to_annotation = dict()
        if 'annotations' in kwargs:
            for key, annotation in kwargs['annotations'].items():
                if key in self.nodes:
                    artist_to_annotation[self.node_artists[key]] = annotation
                elif key in self.edges:
                    artist_to_annotation[self.edge_artists[key]] = annotation
                else:
                    raise ValueError(f"There is no node or edge with the ID {key} for the annotation '{annotation}'.")

        if 'annotation_fontdict' in kwargs:
            AnnotateOnClick.__init__(self, artist_to_annotation, kwargs['annotation_fontdict'])
        else:
            AnnotateOnClick.__init__(self, artist_to_annotation)

        artist_to_table = dict()
        if 'tables' in kwargs:
            for key, table in kwargs['tables'].items():
                if key in self.nodes:
                    artist_to_table[self.node_artists[key]] = table
                elif key in self.edges:
                    artist_to_table[self.edge_artists[key]] = table
                else:
                    raise ValueError(f"There is no node or edge with the ID {key} for the table '{table}'.")

        if 'table_kwargs' in kwargs:
            TableOnClick.__init__(self, artist_to_table, kwargs['table_kwargs'])
        else:
            TableOnClick.__init__(self, artist_to_table)


    def _on_motion(self, event):
        DraggableGraph._on_motion(self, event)
        EmphasizeOnHoverGraph._on_motion(self, event)


    def _on_release(self, event):
        if self._currently_dragging is False:
            DraggableGraph._on_release(self, event)
            if self.artist_to_annotation:
                AnnotateOnClickGraph._on_release(self, event)
            if self.artist_to_table:
                TableOnClickGraph._on_release(self, event)
        else:
            DraggableGraph._on_release(self, event)
            if self.artist_to_annotation:
                self._redraw_annotations(event)


    def _redraw_annotations(self, event):
        if event.inaxes == self.ax:
            for artist in self.annotated_artists:
                self._remove_annotation(artist)
                placement = self._get_annotation_placement(artist)
                self._add_annotation(artist, *placement)
            self.fig.canvas.draw()
