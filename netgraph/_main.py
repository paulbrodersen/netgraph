#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from collections import OrderedDict

BASE_NODE_SIZE = 1e-2
BASE_EDGE_WIDTH = 1e-2

MAX_CLICK_LENGTH = 0.1 # in seconds; anything longer is a drag motion


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

    """

    # Accept a variety of formats and convert to common denominator.
    edge_list, edge_weight, is_directed = parse_graph(graph)

    if edge_weight:

        # If the graph is weighted, we want to visualise the weights using color.
        # Edge width is another popular choice when visualising weighted networks,
        # but if the variance in weights is large, this typically results in less
        # visually pleasing results.
        edge_color  = get_color(edge_weight, cmap=edge_cmap)
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
        node_positions = fruchterman_reingold_layout(edge_list, **kwargs)
    elif len(node_positions) < len(_get_unique_nodes(edge_list)): # some positions are given but not all
        node_positions = fruchterman_reingold_layout(edge_list, pos=node_positions, fixed=node_positions.keys(), **kwargs)

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


def parse_graph(graph):
    """
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

    Returns:
    --------
    edge_list: m-long list of 2-tuples
        List of edges. Each tuple corresponds to an edge defined by (source, target).

    edge_weights: dict (source, target) : float or None
        Edge weights. If the graph is unweighted, None is returned.

    is_directed: bool
        True, if the graph appears to be directed due to
            - the graph object class being passed in (e.g. a networkx.DiGraph), or
            - the existence of bi-directional edges.
    """

    if isinstance(graph, (list, tuple, set)):
        return _parse_sparse_matrix_format(graph)

    elif isinstance(graph, np.ndarray):
        rows, columns = graph.shape
        if columns in (2, 3):
            return _parse_sparse_matrix_format(graph)
        else:
            return _parse_adjacency_matrix(graph)

    # this is terribly unsafe but we don't want to import igraph
    # unless we already know that we need it
    elif str(graph.__class__) == "<class 'igraph.Graph'>":
        return _parse_igraph_graph(graph)

    # ditto
    elif str(graph.__class__) in ("<class 'networkx.classes.graph.Graph'>",
                                  "<class 'networkx.classes.digraph.DiGraph'>",
                                  "<class 'networkx.classes.multigraph.MultiGraph'>",
                                  "<class 'networkx.classes.multidigraph.MultiDiGraph'>"):
        return _parse_networkx_graph(graph)

    else:
        allowed = ['list', 'tuple', 'set', 'networkx.Graph', 'igraph.Graph']
        raise NotImplementedError("Input graph must be one of: {}\nCurrently, type(graph) = {}".format("\n\n\t" + "\n\t".join(allowed)), type(graph))


def _parse_edge_list(edge_list):
    # Edge list may be an array, or a list of lists.
    # We want a list of tuples.
    return [(source, target) for (source, target) in edge_list]


def _parse_sparse_matrix_format(adjacency):
    adjacency = np.array(adjacency)
    rows, columns = adjacency.shape
    if columns == 2:
        edge_list = _parse_edge_list(adjacency)
        return edge_list, None, _is_directed(edge_list)
    elif columns == 3:
        edge_list = _parse_edge_list(adjacency[:,:2])
        edge_weights = {(source, target) : weight for (source, target, weight) in adjacency}

        # In a sparse adjacency format with weights,
        # the type of nodes is promoted to the same type as weights,
        # which is commonly a float. If all nodes can safely be demoted to ints,
        # then we probably want to do that.
        tmp = [(_save_cast_float_to_int(source), _save_cast_float_to_int(target)) for (source, target) in edge_list]
        if np.all([isinstance(num, int) for num in _flatten(tmp)]):
            edge_list = tmp

        if len(set(edge_weights.values())) > 1:
            return edge_list, edge_weights, _is_directed(edge_list)
        else:
            return edge_list, None, _is_directed(edge_list)
    else:
        raise ValueError("Graph specification in sparse matrix format needs to consist of an iterable of tuples of length 2 or 3. Got iterable of tuples of length {}.".format(columns))


def _save_cast_float_to_int(num):
    if np.isclose(num, int(num)):
        return int(num)
    else:
        return num


def _flatten(nested_list):
    return [item for sublist in nested_list for item in sublist]


def _get_unique_nodes(edge_list):
    """
    Using numpy.unique promotes nodes to numpy.float/numpy.int/numpy.str,
    and breaks for nodes that have a more complicated type such as a tuple.
    """
    return list(set(_flatten(edge_list)))


def _parse_adjacency_matrix(adjacency):
    sources, targets = np.where(adjacency)
    edge_list = list(zip(sources.tolist(), targets.tolist()))
    edge_weights = {(source, target): adjacency[source, target] for (source, target) in edge_list}
    if len(set(list(edge_weights.values()))) == 1:
        return edge_list, None, _is_directed(edge_list)
    else:
        return edge_list, edge_weights, _is_directed(edge_list)


def _parse_networkx_graph(graph, attribute_name='weight'):
    edge_list = list(graph.edges())
    try:
        edge_weights = {edge : graph.get_edge_data(*edge)[attribute_name] for edge in edge_list}
    except KeyError: # no weights
        edge_weights = None
    return edge_list, edge_weights, graph.is_directed()


def _parse_igraph_graph(graph):
    edge_list = [(edge.source, edge.target) for edge in graph.es()]
    if graph.is_weighted():
        edge_weights = {(edge.source, edge.target) : edge['weight'] for edge in graph.es()}
    else:
        edge_weights = None
    return edge_list, edge_weights, graph.is_directed()


def get_color(mydict, cmap='RdGy', vmin=None, vmax=None):
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
    values = np.array(list(mydict.values()))

    # apply edge_vmin, edge_vmax
    if vmin:
        values[values<vmin] = vmin

    if vmax:
        values[values>vmax] = vmax

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
    # reorder plot elements such that darker items are plotted last
    # and hence most prominent in the graph
    zorder = np.argsort(np.sum(list(color_dict.values()), axis=1)) # assumes RGB specification
    zorder = np.max(zorder) - zorder # reverse order as greater values correspond to lighter colors
    zorder = {key: index for key, index in zip(color_dict.keys(), zorder)}
    return zorder


def _is_directed(edge_list):
    # test for bi-directional edges
    for (source, target) in edge_list:
        if ((target, source) in edge_list) and (source != target):
            return True
    return False


def _find_renderer(fig):
    """
    https://stackoverflow.com/questions/22667224/matplotlib-get-text-bounding-box-independent-of-backend
    """

    if hasattr(fig.canvas, "get_renderer"):
        #Some backends, such as TkAgg, have the get_renderer method, which
        #makes this easy.
        renderer = fig.canvas.get_renderer()
    else:
        #Other backends do not have the get_renderer method, so we have a work
        #around to find the renderer.  Print the figure to a temporary file
        #object, and then grab the renderer that was used.
        #(I stole this trick from the matplotlib backend_bases.py
        #print_figure() method.)
        import io
        fig.canvas.print_pdf(io.BytesIO())
        renderer = fig._cachedRenderer
    return(renderer)


def _get_text_object_dimenstions(ax, string, *args, **kwargs):
    text_object = ax.text(0., 0., string, *args, **kwargs)
    renderer = _find_renderer(text_object.get_figure())
    bbox_in_display_coordinates = text_object.get_window_extent(renderer)
    bbox_in_data_coordinates = bbox_in_display_coordinates.transformed(ax.transData.inverted())
    w, h = bbox_in_data_coordinates.width, bbox_in_data_coordinates.height
    text_object.remove()
    return w, h


def _get_font_size(ax, node_labels, **kwargs):
    """
    Determine the maximum font size that results in labels that still all fit inside the node face artist.

    TODO:
    -----
    - add font / fontfamily as optional argument
    - potentially, return a dictionary of font sizes instead; then rescale font sizes individually on a per node basis
    """

    # check if there are node sizes or edge widths in kwargs
    if 'node_size' in kwargs:
        node_size = kwargs['node_size']
    else:
        node_size = 3. # default

    if 'node_edge_width' in kwargs:
        node_edge_width = kwargs['node_edge_width']
    else:
        node_edge_width = 0.5 # default

    # initialise base values for font size and rescale factor
    default_font_size = 12.
    rescale_factor = np.nan
    widest = 0.

    # find widest node label; use its rescale factor to set font size for all labels
    for key, label in node_labels.items():

        if isinstance(node_size, (int, float)):
            r = node_size
        elif isinstance(node_size, dict):
            r = node_size[key]

        if isinstance(node_edge_width, (int, float)):
            e = node_edge_width
        elif isinstance(node_edge_width, dict):
            e = node_edge_width[key]

        d = 2 * (r-e) * BASE_NODE_SIZE

        width, height = _get_text_object_dimenstions(ax, label, size=default_font_size)

        if width > widest:
            widest = width
            rescale_factor = d / np.sqrt(width**2 + height**2)

    font_size = default_font_size * rescale_factor

    return font_size


def draw_nodes(node_positions,
               node_shape='o',
               node_size=3.,
               node_edge_width=0.5,
               node_color='w',
               node_edge_color='k',
               node_alpha=1.0,
               node_edge_alpha=1.0,
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
       NOTE: Value is rescaled by BASE_NODE_SIZE (1e-2) to work well with layout routines in igraph and networkx.

    node_edge_width : scalar or dict key : float (default 0.5)
       Line width of node marker border.
       NOTE: Value is rescaled by BASE_NODE_SIZE (1e-2) to work well with layout routines in igraph and networkx.

    node_color : matplotlib color specification or dict node : color specification (default 'w')
       Node color.

    node_edge_color : matplotlib color specification or dict node : color specification (default 'k')
       Node edge color.

    node_alpha : scalar or dict node : float (default 1.)
       The node transparency.

    node_edge_alpha : scalar or dict node : float (default 1.)
       The node edge transparency.

    ax : matplotlib.axis instance or None (default None)
       Axis to plot onto; if none specified, one will be instantiated with plt.gca().

    Returns
    -------
    node_faces: dict node : artist
        Mapping of nodes to the node face artists.

    node_edges: dict node : artist
        Mapping of nodes to the node edge artists.

    """

    if ax is None:
        ax = plt.gca()

    # convert all inputs to dicts mapping node:property
    nodes = node_positions.keys()
    number_of_nodes = len(nodes)

    if isinstance(node_size, (int, float)):
        node_size = {node:node_size for node in nodes}
    if isinstance(node_edge_width, (int, float)):
        node_edge_width = {node: node_edge_width for node in nodes}

    # Simulate node edge by drawing a slightly larger node artist.
    # I wish there was a better way to do this,
    # but this seems to be the only way to guarantee constant proportions,
    # as linewidth argument in matplotlib.patches will not be proportional
    # to a given node radius.
    node_edges = _draw_nodes(node_positions,
                             node_shape=node_shape,
                             node_size=node_size,
                             node_color=node_edge_color,
                             node_alpha=node_edge_alpha,
                             ax=ax,
                             **kwargs)

    node_size = {node: node_size[node] - node_edge_width[node] for node in nodes}
    node_faces = _draw_nodes(node_positions,
                             node_shape=node_shape,
                             node_size=node_size,
                             node_color=node_color,
                             node_alpha=node_alpha,
                             ax=ax,
                             **kwargs)

    return node_faces, node_edges


def _draw_nodes(node_positions,
                node_shape='o',
                node_size=3.,
                node_color='r',
                node_alpha=1.0,
                ax=None,
                **kwargs):
    """
    Draw node markers at specified positions.

    Arguments
    ----------
    node_positions : dict node : (float, float)
        Mapping of nodes to (x, y) positions

    node_shape : string or dict key : string (default 'o')
       The shape of the node. Specification is as for matplotlib.scatter
       marker, i.e. one of 'so^>v<dph8'.
       If a single string is provided all nodes will have the same shape.

    node_size : scalar or dict node : float (default 3.)
       Size (radius) of nodes.
       NOTE: Value is rescaled by BASE_NODE_SIZE (1e-2) to work well with layout routines in igraph and networkx.

    node_color : matplotlib color specification or dict node : color specification (default 'w')
       Node color.

    node_alpha : scalar or dict node : float (default 1.)
       The node transparency.

    ax : matplotlib.axis instance or None (default None)
       Axis to plot onto; if none specified, one will be instantiated with plt.gca().

    Returns
    -------
    artists: dict node : artist
        Mapping of nodes to the artists,

    """

    if ax is None:
        ax = plt.gca()

    # convert all inputs to dicts mapping node:property
    nodes = node_positions.keys()
    number_of_nodes = len(nodes)

    if isinstance(node_shape, str):
        node_shape = {node:node_shape for node in nodes}
    if not isinstance(node_color, dict):
        node_color = {node:node_color for node in nodes}
    if isinstance(node_alpha, (int, float)):
        node_alpha = {node:node_alpha for node in nodes}

    # rescale
    node_size = {node: size  * BASE_NODE_SIZE for (node, size)  in node_size.items()}

    artists = dict()
    for node in nodes:
        node_artist = _get_node_artist(shape=node_shape[node],
                                       position=node_positions[node],
                                       size=node_size[node],
                                       facecolor=node_color[node],
                                       alpha=node_alpha[node],
                                       zorder=2)

        # add artists to axis
        ax.add_artist(node_artist)

        # return handles to artists
        artists[node] = node_artist

    return artists


def _get_node_artist(shape, position, size, facecolor, alpha, zorder=2):
    if shape == 'o': # circle
        artist = matplotlib.patches.Circle(xy=position,
                                           radius=size,
                                           facecolor=facecolor,
                                           alpha=alpha,
                                           linewidth=0.,
                                           zorder=zorder)
    elif shape == '^': # triangle up
        artist = matplotlib.patches.RegularPolygon(xy=position,
                                                   radius=size,
                                                   numVertices=3,
                                                   facecolor=facecolor,
                                                   alpha=alpha,
                                                   orientation=0,
                                                   linewidth=0.,
                                                   zorder=zorder)
    elif shape == '<': # triangle left
        artist = matplotlib.patches.RegularPolygon(xy=position,
                                                   radius=size,
                                                   numVertices=3,
                                                   facecolor=facecolor,
                                                   alpha=alpha,
                                                   orientation=np.pi*0.5,
                                                   linewidth=0.,
                                                   zorder=zorder)
    elif shape == 'v': # triangle down
        artist = matplotlib.patches.RegularPolygon(xy=position,
                                                   radius=size,
                                                   numVertices=3,
                                                   facecolor=facecolor,
                                                   alpha=alpha,
                                                   orientation=np.pi,
                                                   linewidth=0.,
                                                   zorder=zorder)
    elif shape == '>': # triangle right
        artist = matplotlib.patches.RegularPolygon(xy=position,
                                                   radius=size,
                                                   numVertices=3,
                                                   facecolor=facecolor,
                                                   alpha=alpha,
                                                   orientation=np.pi*1.5,
                                                   linewidth=0.,
                                                   zorder=zorder)
    elif shape == 's': # square
        artist = matplotlib.patches.RegularPolygon(xy=position,
                                                   radius=size,
                                                   numVertices=4,
                                                   facecolor=facecolor,
                                                   alpha=alpha,
                                                   orientation=np.pi*0.25,
                                                   linewidth=0.,
                                                   zorder=zorder)
    elif shape == 'd': # diamond
        artist = matplotlib.patches.RegularPolygon(xy=position,
                                                   radius=size,
                                                   numVertices=4,
                                                   facecolor=facecolor,
                                                   alpha=alpha,
                                                   orientation=np.pi*0.5,
                                                   linewidth=0.,
                                                   zorder=zorder)
    elif shape == 'p': # pentagon
        artist = matplotlib.patches.RegularPolygon(xy=position,
                                                   radius=size,
                                                   numVertices=5,
                                                   facecolor=facecolor,
                                                   alpha=alpha,
                                                   linewidth=0.,
                                                   zorder=zorder)
    elif shape == 'h': # hexagon
        artist = matplotlib.patches.RegularPolygon(xy=position,
                                                   radius=size,
                                                   numVertices=6,
                                                   facecolor=facecolor,
                                                   alpha=alpha,
                                                   linewidth=0.,
                                                   zorder=zorder)
    elif shape == 8: # octagon
        artist = matplotlib.patches.RegularPolygon(xy=position,
                                                   radius=size,
                                                   numVertices=8,
                                                   facecolor=facecolor,
                                                   alpha=alpha,
                                                   linewidth=0.,
                                                   zorder=zorder)
    else:
        raise ValueError("Node shape one of: ''so^>v<dph8'. Current shape:{}".format(shape))

    return artist


def draw_edges(edge_list,
               node_positions,
               node_size=3.,
               edge_width=1.,
               edge_color='k',
               edge_alpha=1.,
               edge_zorder=None,
               draw_arrows=True,
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
        NOTE: Value is rescaled by BASE_EDGE_WIDTH (1e-2) to work well with layout routines in igraph and networkx.

    edge_color : matplotlib color specification or
                 dict (source, target) : color specification (default 'k')
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

    ax : matplotlib.axis instance or None (default None)
       Axis to plot onto; if none specified, one will be instantiated with plt.gca().

    Returns
    -------
    artists: dict (source, target) : artist
        Mapping of edges to matplotlib.patches.FancyArrow artists.

    """

    if ax is None:
        ax = plt.gca()

    edge_list = _parse_edge_list(edge_list)
    nodes = node_positions.keys()
    number_of_nodes = len(nodes)

    if isinstance(node_size, (int, float)):
        node_size = {node:node_size for node in nodes}
    if isinstance(edge_width, (int, float)):
        edge_width = {edge: edge_width for edge in edge_list}
    if not isinstance(edge_color, dict):
        edge_color = {edge: edge_color for edge in edge_list}
    if isinstance(edge_alpha, (int, float)):
        edge_alpha = {edge: edge_alpha for edge in edge_list}

    # rescale
    node_size  = {node: size  * BASE_NODE_SIZE  for (node, size)  in node_size.items()}
    edge_width = {edge: width * BASE_EDGE_WIDTH for (edge, width) in edge_width.items()}

    # order edges if necessary
    if not (edge_zorder is None):
       edge_list = sorted(edge_zorder, key=lambda k: edge_zorder[k])

    # NOTE: At the moment, only the relative zorder is honored, not the absolute value.

    artists = dict()
    for (source, target) in edge_list:

        if source != target:

            x1, y1 = node_positions[source]
            x2, y2 = node_positions[target]

            dx = x2-x1
            dy = y2-y1

            width = edge_width[(source, target)]
            color = edge_color[(source, target)]
            alpha = edge_alpha[(source, target)]

            if (target, source) in edge_list: # i.e. bidirectional
                # shift edge to the right (looking along the arrow)
                x1, y1, x2, y2 = _shift_edge(x1, y1, x2, y2, delta=0.5*width)
                # plot half arrow / line
                shape = 'right'
            else:
                shape = 'full'

            if draw_arrows:
                offset = node_size[target]
                head_length = 2 * width
                head_width = 3 * width
                length_includes_head = True
            else:
                offset = None
                head_length = 1e-10 # 0 throws error
                head_width = 1e-10 # 0 throws error
                length_includes_head = False

            patch = FancyArrow(x1, y1, dx, dy,
                               width=width,
                               facecolor=color,
                               head_length=head_length,
                               head_width=head_width,
                               length_includes_head=length_includes_head,
                               zorder=1,
                               edgecolor='none',
                               linewidth=0.1,
                               offset=offset,
                               shape=shape)
            ax.add_artist(patch)
            artists[(source, target)] = patch

        else: # source == target, i.e. a self-loop
            import warnings
            warnings.warn("Plotting of self-loops not supported. Ignoring edge ({}, {}).".format(source, target))

    return artists


def _shift_edge(x1, y1, x2, y2, delta):
    # get orthogonal unit vector
    v = np.r_[x2-x1, y2-y1] # original
    v = np.r_[-v[1], v[0]] # orthogonal
    v = v / np.linalg.norm(v) # unit
    dx, dy = delta * v
    return x1+dx, y1+dy, x2+dx, y2+dy


class FancyArrow(matplotlib.patches.Polygon):
    """
    This is an expansion of of matplotlib.patches.FancyArrow.
    """

    _edge_default = True

    def __str__(self):
        return "FancyArrow()"

    def __init__(self, x, y, dx, dy, width=0.001, length_includes_head=False,
                 head_width=None, head_length=None, shape='full', overhang=0,
                 head_starts_at_zero=False, offset=None, **kwargs):
        """
        Constructor arguments
          *width*: float (default: 0.001)
            width of full arrow tail

          *length_includes_head*: [True | False] (default: False)
            True if head is to be counted in calculating the length.

          *head_width*: float or None (default: 3*width)
            total width of the full arrow head

          *head_length*: float or None (default: 1.5 * head_width)
            length of arrow head

          *shape*: ['full', 'left', 'right'] (default: 'full')
            draw the left-half, right-half, or full arrow

          *overhang*: float (default: 0)
            fraction that the arrow is swept back (0 overhang means
            triangular shape). Can be negative or greater than one.

          *head_starts_at_zero*: [True | False] (default: False)
            if True, the head starts being drawn at coordinate 0
            instead of ending at coordinate 0.

        Other valid kwargs (inherited from :class:`Patch`) are:
        %(Patch)s

        """
        self.width = width

        if head_width is None:
            self.head_width = 3 * self.width
        else:
            self.head_width = head_width

        if head_length is None:
            self.head_length = 1.5 * self.head_width
        else:
            self.head_length = head_length

        self.length_includes_head = length_includes_head
        self.head_starts_at_zero = head_starts_at_zero
        self.overhang = overhang
        self.shape = shape
        self.offset = offset

        verts = self.compute_vertices(x, y, dx, dy)

        matplotlib.patches.Polygon.__init__(self, list(map(tuple, verts)), closed=True, **kwargs)

    def compute_vertices(self, x, y, dx, dy):

        distance = np.hypot(dx, dy)

        if self.offset:
            dx *= (distance-self.offset)/distance
            dy *= (distance-self.offset)/distance
            distance = np.hypot(dx, dy)
            # distance -= self.offset

        if self.length_includes_head:
            length = distance
        else:
            length = distance + self.head_length
        if not length:
            verts = []  # display nothing if empty
        else:
            # start by drawing horizontal arrow, point at (0,0)
            hw, hl, hs, lw = self.head_width, self.head_length, self.overhang, self.width
            left_half_arrow = np.array([
                [0.0, 0.0],                   # tip
                [-hl, -hw / 2.0],             # leftmost
                [-hl * (1 - hs), -lw / 2.0],  # meets stem
                [-length, -lw / 2.0],         # bottom left
                [-length, 0],
            ])
            # if we're not including the head, shift up by head length
            if not self.length_includes_head:
                left_half_arrow += [self.head_length, 0]
            # if the head starts at 0, shift up by another head length
            if self.head_starts_at_zero:
                left_half_arrow += [self.head_length / 2.0, 0]
            # figure out the shape, and complete accordingly
            if self.shape == 'left':
                coords = left_half_arrow
            else:
                right_half_arrow = left_half_arrow * [1, -1]
                if self.shape == 'right':
                    coords = right_half_arrow
                elif self.shape == 'full':
                    # The half-arrows contain the midpoint of the stem,
                    # which we can omit from the full arrow. Including it
                    # twice caused a problem with xpdf.
                    coords = np.concatenate([left_half_arrow[:-1],
                                             right_half_arrow[-1::-1]])
                else:
                    raise ValueError("Got unknown shape: %s" % self.shape)
            if distance != 0:
                cx = float(dx) / distance
                sx = float(dy) / distance
            else:
                #Account for division by zero
                cx, sx = 0, 1
            M = np.array([[cx, sx], [-sx, cx]])
            verts = np.dot(coords, M) + (x + dx, y + dy)

        return verts


    def update_vertices(self, x0, y0, dx, dy):
        verts = self.compute_vertices(x0, y0, dx, dy)
        self.set_xy(verts)


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
        NOTE: Value is rescaled by BASE_EDGE_WIDTH (1e-2) to work well with layout routines in igraph and networkx.

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

    edge_width = {edge: width * BASE_EDGE_WIDTH for (edge, width) in edge_width.items()}

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
                angle = np.arctan2(y2-y1, x2-x1)/(2.0*np.pi)*360  # degrees
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
            import warnings
            warnings.warn("Plotting of edge labels for self-loops not supported. Ignoring edge with label: {}".format(label))

    return text_items


def _update_view(node_positions, ax, node_size=3.):
    # Pad x and y limits as patches are not registered properly
    # when matplotlib sets axis limits automatically.
    # Hence we need to set them manually.

    if isinstance(node_size, dict):
        maxs = np.max(list(node_size.values())) * BASE_NODE_SIZE
    else:
        maxs = node_size * BASE_NODE_SIZE

    maxx, maxy = np.max(list(node_positions.values()), axis=0)
    minx, miny = np.min(list(node_positions.values()), axis=0)

    w = maxx-minx
    h = maxy-miny
    padx, pady = 0.05*w + maxs, 0.05*h + maxs
    corners = (minx-padx, miny-pady), (maxx+padx, maxy+pady)

    ax.update_datalim(corners)
    ax.autoscale_view()
    ax.get_figure().canvas.draw()


def _make_pretty(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.get_figure().set_facecolor('w')
    ax.set_frame_on(False)
    ax.get_figure().canvas.draw()


# --------------------------------------------------------------------------------
# Spring layout


def fruchterman_reingold_layout(edge_list,
                                edge_weights=None,
                                k=None,
                                pos=None,
                                fixed=None,
                                iterations=50,
                                scale=1,
                                center=np.zeros((2)),
                                dim=2,
                                **kwargs):
    """
    Position nodes using Fruchterman-Reingold force-directed algorithm.

    Parameters
    ----------
    edge_list: m-long iterable of 2-tuples or equivalent (such as (m, 2) ndarray)
        List of edges. Each tuple corresponds to an edge defined by (source, target).

    edge_weights: dict (source, target) : float or None (default=None)
        Edge weights.

    k : float (default=None)
        Optimal distance between nodes.  If None the distance is set to
        1/sqrt(n) where n is the number of nodes.  Increase this value
        to move nodes farther apart.

    pos : dict or None  optional (default=None)
        Initial positions for nodes as a dictionary with node as keys
        and values as a coordinate list or tuple.  If None, then use
        random initial positions.

    fixed : list or None  optional (default=None)
        Nodes to keep fixed at initial position.

    iterations : int  optional (default=50)
        Number of iterations of spring-force relaxation

    scale : number (default: 1)
        Scale factor for positions. Only used if `fixed is None`.

    center : array-like or None
        Coordinate pair around which to center the layout.
        Only used if `fixed is None`.

    dim : int
        Dimension of layout.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node

    Notes:
    ------
    Implementation taken with minor modifications from networkx.spring_layout().

    """

    nodes = _get_unique_nodes(edge_list)
    total_nodes = len(nodes)

    # translate fixed node ID to position in node list
    if fixed is not None:
        node_to_idx = dict(zip(nodes, range(total_nodes)))
        fixed = np.asarray([node_to_idx[v] for v in fixed])

    if pos is not None:
        # Determine size of existing domain to adjust initial positions
        domain_size = max(coord for pos_tup in pos.values() for coord in pos_tup)
        if domain_size == 0:
            domain_size = 1
        shape = (total_nodes, dim)
        pos_arr = np.random.random(shape) * domain_size + center
        for i, n in enumerate(nodes):
            if n in pos:
                pos_arr[i] = np.asarray(pos[n])
    else:
        pos_arr = None

    if k is None and fixed is not None:
        # We must adjust k by domain size for layouts not near 1x1
        k = domain_size / np.sqrt(total_nodes)

    A = _edge_list_to_adjacency(edge_list, edge_weights)
    pos = _dense_fruchterman_reingold(A, k, pos_arr, fixed, iterations, dim)

    if fixed is None:
        pos = _rescale_layout(pos, scale=scale) + center

    return dict(zip(nodes, pos))


spring_layout = fruchterman_reingold_layout


def _dense_fruchterman_reingold(A, k=None, pos=None, fixed=None,
                                iterations=50, dim=2):
    """
    Position nodes in adjacency matrix A using Fruchterman-Reingold
    """

    nnodes, _ = A.shape

    # if pos is None:
    #     # random initial positions
    #     pos = np.asarray(np.random.random((nnodes, dim)), dtype=A.dtype)
    # else:
    #     # make sure positions are of same type as matrix
    #     pos = pos.astype(A.dtype)

    if pos is None:
        # random initial positions
        pos = np.random.rand(nnodes, dim)

    # optimal distance between nodes
    if k is None:
        k = np.sqrt(1.0/nnodes)
    # the initial "temperature"  is about .1 of domain area (=1x1)
    # this is the largest step allowed in the dynamics.
    # We need to calculate this in case our fixed positions force our domain
    # to be much bigger than 1x1
    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1]))*0.1
    # simple cooling scheme.
    # linearly step down by dt on each iteration so last iteration is size dt.
    dt = t/float(iterations+1)
    delta = np.zeros((pos.shape[0], pos.shape[0], pos.shape[1]), dtype=A.dtype)
    # the inscrutable (but fast) version
    # this is still O(V^2)
    # could use multilevel methods to speed this up significantly
    for iteration in range(iterations):
        # matrix of difference between points
        delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        # distance between points
        distance = np.linalg.norm(delta, axis=-1)
        # enforce minimum distance of 0.01
        np.clip(distance, 0.01, None, out=distance)
        # displacement "force"
        displacement = np.einsum('ijk,ij->ik',
                                 delta,
                                 (k * k / distance**2 - A * distance / k))
        # update positions
        length = np.linalg.norm(displacement, axis=-1)
        length = np.where(length < 0.01, 0.1, length)
        delta_pos = np.einsum('ij,i->ij', displacement, t / length)
        if fixed is not None:
            # don't change positions of fixed nodes
            delta_pos[fixed] = 0.0
        pos += delta_pos
        # cool temperature
        t -= dt
    return pos


def _rescale_layout(pos, scale=1):
    """Return scaled position array to (-scale, scale) in all axes.

    The function acts on NumPy arrays which hold position information.
    Each position is one row of the array. The dimension of the space
    equals the number of columns. Each coordinate in one column.

    To rescale, the mean (center) is subtracted from each axis separately.
    Then all values are scaled so that the largest magnitude value
    from all axes equals `scale` (thus, the aspect ratio is preserved).
    The resulting NumPy Array is returned (order of rows unchanged).

    Parameters
    ----------
    pos : numpy array
        positions to be scaled. Each row is a position.

    scale : number (default: 1)
        The size of the resulting extent in all directions.

    Returns
    -------
    pos : numpy array
        scaled positions. Each row is a position.

    """
    # Find max length over all dimensions
    lim = 0  # max coordinate for all axes
    for i in range(pos.shape[1]):
        pos[:, i] -= pos[:, i].mean()
        lim = max(abs(pos[:, i]).max(), lim)
    # rescale to (-scale, scale) in all directions, preserves aspect
    if lim > 0:
        for i in range(pos.shape[1]):
            pos[:, i] *= scale / lim
    return pos


def _edge_list_to_adjacency(edge_list, edge_weights=None):

    sources = [s for (s, _) in edge_list]
    targets = [t for (_, t) in edge_list]

    if edge_weights:
        weights = [edge_weights[edge] for edge in edge_list]
    else:
        weights = np.ones((len(edge_list)))

    # map nodes to consecutive integers
    nodes = sources + targets
    unique, indices = np.unique(nodes, return_inverse=True)
    node_to_idx = dict(zip(unique, indices))
    source_indices = [node_to_idx[source] for source in sources]
    target_indices = [node_to_idx[target] for target in targets]

    total_nodes = len(unique)
    adjacency_matrix = np.zeros((total_nodes, total_nodes))
    adjacency_matrix[source_indices, target_indices] = weights

    return adjacency_matrix


# --------------------------------------------------------------------------------
# interactive plotting

class Graph(object):

    def __init__(self, graph, node_positions=None, node_labels=None, edge_labels=None, edge_cmap='RdGy', ax=None, **kwargs):
        """
        Initialises the Graph object.
        Upon initialisation, it will try to do "the right thing".

        For finer control of the individual draw elements,
        and a complete list of keyword arguments, see the class methods:

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

        """

        self.draw(graph, node_positions, node_labels, edge_labels, edge_cmap, ax, **kwargs)


    def draw(self, graph, node_positions=None, node_labels=None, edge_labels=None, edge_cmap='RdGy', ax=None, **kwargs):
        """
        Initialises / redraws the Graph object.
        Upon initialisation, it will try to do "the right thing".

        For finer control of the individual draw elements,
        and a complete list of keyword arguments, see the class methods:

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

        """

        # --------------------------------------------------------------------------------
        # TODO: split off / move to __init__ (potentially)

        # Create axis if none is given.
        if ax is None:
            self.ax = plt.gca()
        else:
            self.ax = ax

        # Accept a variety of formats for 'graph' and convert to common denominator.
        self.edge_list, self.edge_weight, is_directed = parse_graph(graph)

        # Color and reorder edges for weighted graphs.
        if self.edge_weight:

            # If the graph is weighted, we want to visualise the weights using color.
            # Edge width is another popular choice when visualising weighted networks,
            # but if the variance in weights is large, this typically results in less
            # visually pleasing results.
            self.edge_color  = get_color(self.edge_weight, cmap=edge_cmap)
            kwargs.setdefault('edge_color', self.edge_color)

            # Plotting darker edges over lighter edges typically results in visually
            # more pleasing results. Here we hence specify the relative order in
            # which edges are plotted according to the color of the edge.
            self.edge_zorder = _get_zorder(self.edge_color)
            kwargs.setdefault('edge_zorder', self.edge_zorder)

        # --------------------------------------------------------------------------------

        # Plot arrows if the graph has bi-directional edges.
        if is_directed:
            kwargs.setdefault('draw_arrows', True)
        else:
            kwargs.setdefault('draw_arrows', False)

        # Initialise node positions.
        if node_positions is None:
            # If none are given, initialise all.
            self.node_positions = self._get_node_positions(self.edge_list)
        elif len(node_positions) < len(_get_unique_nodes(self.edge_list)):
            # If some are given, keep those fixed and initialise remaining.
            self.node_positions = self._get_node_positions(self.edge_list, pos=node_positions, fixed=node_positions.keys(), **kwargs)
        else:
            # If all are given, don't do anything.
            self.node_positions = node_positions

        # Draw plot elements.
        self.draw_edges(self.edge_list, self.node_positions, ax=self.ax, **kwargs)
        self.draw_nodes(self.node_positions, ax=self.ax, **kwargs)

        # Improve default layout of axis.
        # This function needs to be called before any font sizes are adjusted,
        # as the axis dimensions affect the effective font size.
        self._update_view()

        if node_labels:
            if not hasattr(self, 'node_labels'):
                self.node_labels = node_labels
            else:
                self.node_labels.update(node_labels)

            if not 'node_label_font_size' in kwargs:
                # set font size such that even the largest label fits inside node label face artist
                self.node_label_font_size = _get_font_size(self.ax, self.node_labels, **kwargs) * 0.9 # conservative fudge factor
                self.draw_node_labels(self.node_labels, self.node_positions, node_label_font_size=self.node_label_font_size, ax=self.ax, **kwargs)
            else:
                self.draw_node_labels(self.node_labels, self.node_positions, ax=self.ax, **kwargs)

        if edge_labels:
            if not hasattr(self, 'edge_labels'):
                self.edge_labels = edge_labels
            else:
                self.edge_labels.update(edge_labels)
            self.draw_edge_labels(self.edge_list, self.edge_labels, self.node_positions, ax=self.ax, **kwargs)

        _make_pretty(self.ax)


    def _get_node_positions(self, *args, **kwargs):
        """
        Ultra-thin wrapper around fruchterman_reingold_layout.
        Allows method to be overwritten by derived classes.
        """
        return fruchterman_reingold_layout(*args, **kwargs)


    def draw_nodes(self, *args, **kwargs):
        """
        Draw node markers at specified positions.

        Arguments
        ----------
        node_positions : dict node : (float, float)
            Mapping of nodes to (x, y) positions

        node_shape : string or dict key : string (default 'o')
           The shape of the node. Specification is as for matplotlib.scatter
           marker, i.e. one of 'so^>v<dph8'.
           If a single string is provided all nodes will have the same shape.

        node_size : scalar or dict node : float (default 3.)
           Size (radius) of nodes.
           NOTE: Value is rescaled by BASE_NODE_SIZE (1e-2) to work well with layout routines in igraph and networkx.

        node_edge_width : scalar or dict key : float (default 0.5)
           Line width of node marker border.

        node_color : matplotlib color specification or dict node : color specification (default 'w')
           Node color.

        node_edge_color : matplotlib color specification or dict node : color specification (default 'k')
           Node edge color.

        node_alpha : scalar or dict node : float (default 1.)
           The node transparency.

        node_edge_alpha : scalar or dict node : float (default 1.)
           The node edge transparency.

        ax : matplotlib.axis instance or None (default None)
           Axis to plot onto; if none specified, one will be instantiated with plt.gca().

        Updates
        -------
        self.node_face_artists: dict node : artist
            Mapping of nodes to the node face artists.

        self.node_edge_artists: dict node : artist
            Mapping of nodes to the node edge artists.

        """
        node_faces, node_edges = draw_nodes(*args, **kwargs)

        if not hasattr(self, 'node_face_artists'):
            self.node_face_artists = node_faces
        else:
            for key, artist in node_faces.items():
                if key in self.node_face_artists:
                    # remove old artist
                    self.node_face_artists[key].remove()
                # assign new one
                self.node_face_artists[key] = artist

        if not hasattr(self, 'node_edge_artists'):
            self.node_edge_artists = node_edges
        else:
            for key, artist in node_edges.items():
                if key in self.node_edge_artists:
                    # remove old artist
                    self.node_edge_artists[key].remove()
                # assign new one
                self.node_edge_artists[key] = artist


    def draw_edges(self, *args, **kwargs):
        """

        Draw the edges of the network.

        Arguments
        ----------
        edge_list: m-long iterable of 2-tuples or equivalent (such as (m, 2) ndarray)
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
            NOTE: Value is rescaled by BASE_EDGE_WIDTH (1e-2) to work well with layout routines in igraph and networkx.

        edge_color : matplotlib color specification or
                     dict (source, target) : color specification (default 'k')
           Edge color.

        edge_alpha : float or dict (source, target) : float (default 1.)
            The edge transparency,

        edge_zorder: int or dict (source, target) : int (default None)
            Order in which to plot the edges.
            If None, the edges will be plotted in the order they appear in 'adjacency'.
            Note: graphs typically appear more visually pleasing if darker coloured edges
            are plotted on top of lighter coloured edges.

        draw_arrows : bool, optional (default True)
            If True, draws edges with arrow heads.

        ax : matplotlib.axis instance or None (default None)
           Axis to plot onto; if none specified, one will be instantiated with plt.gca().

        Updates
        -------
        self.edge_artists: dict (source, target) : artist
            Mapping of edges to matplotlib.patches.FancyArrow artists.

        """
        artists = draw_edges(*args, **kwargs)

        if not hasattr(self, 'edge_artists'):
            self.edge_artists = artists
        else:
            for key, artist in artists.items():
                if key in self.edge_artists:
                    # remove old artist
                    self.edge_artists[key].remove()
                # assign new one
                self.edge_artists[key] = artist


    def draw_node_labels(self, *args, **kwargs):
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

        clip_on : bool (default False)
           Turn on clipping at axis boundaries.

        ax : matplotlib.axis instance or None (default None)
           Axis to plot onto; if none specified, one will be instantiated with plt.gca().


        Updates
        -------
        self.node_label_artists: dict
            Dictionary mapping node indices to text objects.

        @reference
        Borrowed with minor modifications from networkx/drawing/nx_pylab.py

        """

        artists = draw_node_labels(*args, **kwargs)

        if not hasattr(self, 'node_label_artists'):
            self.node_label_artists = artists
        else:
            for key, artist in artists.items():
                if key in self.node_label_artists:
                    # remove old artist
                    self.node_label_artists[key].remove()
                # assign new one
                self.node_label_artists[key] = artist


    def draw_edge_labels(self, *args, **kwargs):
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
                center : 0.5 (default)
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
            NOTE: Value is rescaled by BASE_EDGE_WIDTH (1e-2) to work well with layout routines in igraph and networkx.

        ax : matplotlib.axis instance or None (default None)
           Axis to plot onto; if none specified, one will be instantiated with plt.gca().

        Updates
        -------
        self.edge_label_artists: dict (source, target) : text object
            Mapping of edges to edge label artists.

        @reference
        Borrowed with minor modifications from networkx/drawing/nx_pylab.py

        """
        artists = draw_edge_labels(*args, **kwargs)

        if not hasattr(self, 'edge_label_artists'):
            self.edge_label_artists = artists
        else:
            for key, artist in artists.items():
                if key in self.edge_label_artists:
                    # remove old artist
                    self.edge_label_artists[key].remove()
                # assign new one
                self.edge_label_artists[key] = artist


    def _update_view(self):
        # Pad x and y limits as patches are not registered properly
        # when matplotlib sets axis limits automatically.
        # Hence we need to set them manually.

        max_edge_radius = np.max([artist.radius for artist in self.node_edge_artists.values()])
        max_face_radius = np.max([artist.radius for artist in self.node_face_artists.values()])
        max_radius = np.max([max_edge_radius, max_face_radius])

        maxx, maxy = np.max(list(self.node_positions.values()), axis=0)
        minx, miny = np.min(list(self.node_positions.values()), axis=0)

        w = maxx-minx
        h = maxy-miny
        padx, pady = 0.05*w + max_radius, 0.05*h + max_radius
        corners = (minx-padx, miny-pady), (maxx+padx, maxy+pady)

        self.ax.update_datalim(corners)
        self.ax.autoscale_view()
        self.ax.get_figure().canvas.draw()


class DraggableArtists(object):
    """
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

        self.fig.canvas.mpl_connect('button_press_event',   self._on_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('motion_notify_event',  self._on_motion)
        self.fig.canvas.mpl_connect('key_press_event',      self._on_key_press)
        self.fig.canvas.mpl_connect('key_release_event',    self._on_key_release)

        self._draggable_artists = artists
        self._clicked_artist = None
        self._time_on_key_press = -np.inf
        self._control_is_held = False
        self._currently_selecting = False
        self._currently_dragging = False
        self._selected_artists = []
        self._offset = dict()
        self._base_alpha = dict([(artist, artist.get_alpha()) for artist in artists])

        self._rect = plt.Rectangle((0, 0), 1, 1, linestyle="--", edgecolor="crimson", fill=False)
        self.ax.add_patch(self._rect)
        self._rect.set_visible(False)

        self._x0 = 0
        self._y0 = 0
        self._x1 = 0
        self._y1 = 0

    def _on_press(self, event):

        if event.inaxes:

            # reset rectangle
            self._x0 = event.xdata
            self._y0 = event.ydata
            self._x1 = event.xdata
            self._y1 = event.ydata

            self._clicked_artist = None

            # is the press over some artist
            is_on_artist = False
            for artist in self._draggable_artists:

                if artist.contains(event)[0]:
                    if artist in self._selected_artists:
                        # print("Clicked on previously selected artist.")
                        # Some artists are already selected,
                        # and the user clicked on an already selected artist.
                        # It remains to be seen if the user wants to
                        # 1) start dragging, or
                        # 2) deselect everything else and select only the last selected artist.
                        # Hence we will defer decision until the user releases mouse button.
                        self._clicked_artist = artist
                        self._time_on_key_press = time.time()

                    else:
                        # print("Clicked on new artist.")
                        # the user wants to select artist and drag
                        if not self._control_is_held:
                            self._deselect_all_artists()
                        self._select_artist(artist)

                    # start dragging
                    self._currently_dragging = True
                    self._offset = {artist : artist.center - np.array([event.xdata, event.ydata]) for artist in self._selected_artists}

                    # do not check anything else
                    # NOTE: if two artists are overlapping, only the first one encountered is selected!
                    break

            else:
                # print("Did not click on artist.")
                if not self._control_is_held:
                    self._deselect_all_artists()

                # start window select
                self._currently_selecting = True

        else:
            print("Warning: clicked outside axis limits!")


    def _on_release(self, event):

        if self._currently_selecting:

            # select artists inside window
            for artist in self._draggable_artists:
                if self._is_inside_rect(*artist.center):
                    if self._control_is_held:               # if/else probably superfluouos
                        self._toggle_select_artist(artist)  # as no artists will be selected
                    else:                                   # if control is not held previously
                        self._select_artist(artist)         #

            # stop window selection and draw new state
            self._currently_selecting = False
            self._rect.set_visible(False)
            self.fig.canvas.draw_idle()

        elif self._currently_dragging:

            # If there was just short 'click and release' not a 'click and hold' indicating a drag motion,
            # we need to (toggle) select the clicked artist and deselect everything else.
            if ((time.time() - self._time_on_key_press) < MAX_CLICK_LENGTH) and (self._clicked_artist is not None):
                if self._control_is_held:
                    self._toggle_select_artist(self._clicked_artist)
                else:
                    self._deselect_all_artists()
                    self._select_artist(self._clicked_artist)

            self._currently_dragging = False


    def _on_motion(self, event):
        if event.inaxes:
            if self._currently_dragging:
                self._move(event)
            elif self._currently_selecting:
                self._x1 = event.xdata
                self._y1 = event.ydata
                # add rectangle for selection here
                self._selector_on()


    def _move(self, event):
        cursor_position = np.array([event.xdata, event.ydata])
        for artist in self._selected_artists:
            artist.center = cursor_position + self._offset[artist]
        self.fig.canvas.draw_idle()


    def _on_key_press(self, event):
       if event.key == 'control':
           self._control_is_held = True


    def _on_key_release(self, event):
       if event.key == 'control':
           self._control_is_held = False


    def _is_inside_rect(self, x, y):
        xlim = np.sort([self._x0, self._x1])
        ylim = np.sort([self._y0, self._y1])
        if (xlim[0]<=x) and (x<xlim[1]) and (ylim[0]<=y) and (y<ylim[1]):
            return True
        else:
            return False


    def _toggle_select_artist(self, artist):
        if artist in self._selected_artists:
            self._deselect_artist(artist)
        else:
            self._select_artist(artist)


    def _select_artist(self, artist):
        if not (artist in self._selected_artists):
            alpha = artist.get_alpha()
            try:
                artist.set_alpha(0.5 * alpha)
            except TypeError: # alpha not explicitly set
                artist.set_alpha(0.5)
            self._selected_artists.append(artist)
            self.fig.canvas.draw_idle()


    def _deselect_artist(self, artist):
        if artist in self._selected_artists:
            artist.set_alpha(self._base_alpha[artist])
            self._selected_artists = [a for a in self._selected_artists if not (a is artist)]
            self.fig.canvas.draw_idle()


    def _deselect_all_artists(self):
        for artist in self._selected_artists:
            artist.set_alpha(self._base_alpha[artist])
        self._selected_artists = []
        self.fig.canvas.draw_idle()


    def _selector_on(self):
        self._rect.set_visible(True)
        xlim = np.sort([self._x0, self._x1])
        ylim = np.sort([self._y0, self._y1])
        self._rect.set_xy((xlim[0],ylim[0] ) )
        self._rect.set_width(np.diff(xlim))
        self._rect.set_height(np.diff(ylim))
        self.fig.canvas.draw_idle()


class InteractiveGraph(Graph, DraggableArtists):

    def __init__(self, *args, **kwargs):
        Graph.__init__(self, *args, **kwargs)
        DraggableArtists.__init__(self, self.node_face_artists.values())

        self._node_to_draggable_artist = self.node_face_artists
        self._draggable_artist_to_node = dict(zip(self.node_face_artists.values(), self.node_face_artists.keys()))

        # trigger resize of labels when canvas size changes
        self.fig.canvas.mpl_connect('resize_event', self._on_resize)


    def _move(self, event):

        cursor_position = np.array([event.xdata, event.ydata])

        nodes = []
        for artist in self._selected_artists:
            node = self._draggable_artist_to_node[artist]
            nodes.append(node)
            self.node_positions[node] = cursor_position + self._offset[artist]

        self._update_nodes(nodes)
        self._update_edges(nodes)
        self.fig.canvas.draw_idle()


    def _update_nodes(self, nodes):
        for node in nodes:
            self.node_edge_artists[node].center = self.node_positions[node]
            self.node_face_artists[node].center = self.node_positions[node]
            if hasattr(self, 'node_label_artists'):
                self.node_label_artists[node].set_position(self.node_positions[node])


    def _update_edges(self, nodes):
        # get edges that need to move
        edges = [(source, target) for (source, target) in self.edge_list if (source in nodes) or (target in nodes)]

        # remove self-loops
        edges = [(source, target) for source, target in edges if source != target]

        # move edges to new positions
        for (source, target) in edges:
            x0, y0 = self.node_positions[source]
            x1, y1 = self.node_positions[target]

            # shift edge right if birectional
            # TODO: potentially move shift into FancyArrow (shape='right)
            if (target, source) in edges: # bidirectional
                x0, y0, x1, y1 = _shift_edge(x0, y0, x1, y1, delta=0.5*self.edge_artists[(source, target)].width)

            # update path
            self.edge_artists[(source, target)].update_vertices(x0=x0, y0=y0, dx=x1-x0, dy=y1-y0)

        # move edge labels
        try:
            self._update_edge_labels(edges, self.node_positions)
        except AttributeError: # no edge labels
            pass


    def _update_edge_labels(self, edges, node_positions, edge_label_position=0.5, rotate=True): # TODO: pass 'rotate' properly

        for (n1, n2) in edges:
            (x1, y1) = node_positions[n1]
            (x2, y2) = node_positions[n2]

            if (n2, n1) in edges: # i.e. bidirectional edge
                x1, y1, x2, y2 = _shift_edge(x1, y1, x2, y2, delta=1.5*self.edge_artists[(source, target)].width)

            (x, y) = (x1 * edge_label_position + x2 * (1.0 - edge_label_position),
                      y1 * edge_label_position + y2 * (1.0 - edge_label_position))

            if rotate:
                angle = np.arctan2(y2-y1, x2-x1)/(2.0*np.pi)*360  # degrees
                # make label orientation "right-side-up"
                if angle > 90:
                    angle -= 180
                if angle < - 90:
                    angle += 180
                # transform data coordinate angle to screen coordinate angle
                xy = np.array((x, y))
                trans_angle = self.ax.transData.transform_angles(np.array((angle,)),
                                                                 xy.reshape((1, 2)))[0]
            else:
                trans_angle = 0.0

            self.edge_label_artists[(n1, n2)].set_position((x, y))


    def _on_resize(self, event):
        if hasattr(self, 'node_labels'):
            self.node_label_font_size = _get_font_size(self.ax, self.node_labels) * 0.9 # conservative fudge factor
            self.draw_node_labels(self.node_labels, self.node_positions, node_label_font_size=self.node_label_font_size, ax=self.ax)


# --------------------------------------------------------------------------------
# Test code


def _get_random_weight_matrix(n, p,
                              weighted=True,
                              strictly_positive=False,
                              directed=True,
                              fully_bidirectional=False,
                              allow_self_loops=False,
                              dales_law=False):

    if weighted:
        w = np.random.randn(n, n)
    else:
        w = np.ones((n, n))

    if strictly_positive:
        w = np.abs(w)

    if not directed:
        w = np.triu(w)

    if directed and fully_bidirectional:
        c = np.random.rand(n, n) <= p/2
        c = np.logical_or(c, c.T)
    else:
        c = np.random.rand(n, n) <= p
    w[~c] = 0.

    if dales_law and weighted and not strictly_positive:
        w = np.abs(w) * np.sign(np.random.randn(n))[:,None]

    if not allow_self_loops:
        w -= np.diag(np.diag(w))

    return w


def _get_random_length_node_labels(total_nodes, m=8, sd=4):

    lengths = m + sd * np.random.randn(total_nodes)
    lengths = lengths.astype(np.int)
    lengths[lengths < 1] = 1

    # https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits-in-python
    import random
    import string
    def string_generator(size, chars=string.ascii_lowercase):
        return ''.join(random.choice(chars) for _ in range(size))

    return {ii : string_generator(lengths[ii]) for ii in range(total_nodes)}


def test(n=20, p=0.15,
         directed=True,
         weighted=True,
         strictly_positive=False,
         test_format='sparse',
         show_node_labels=False,
         show_edge_labels=False,
         InteractiveClass=False,
         **kwargs):

    adjacency_matrix = _get_random_weight_matrix(n, p,
                                                 directed=directed,
                                                 strictly_positive=strictly_positive,
                                                 weighted=weighted)

    sources, targets = np.where(adjacency_matrix)
    weights = adjacency_matrix[sources, targets]
    adjacency = np.c_[sources, targets, weights]

    if show_node_labels:
        # node_labels = {node: str(int(node)) for node in np.r_[sources, targets]}
        node_labels = _get_random_length_node_labels(n)
    else:
        node_labels = None

    if show_edge_labels:
        edge_labels = {(edge[0], edge[1]): str(int(ii)) for ii, edge in enumerate(adjacency)}
    else:
        edge_labels = None

    if test_format == "sparse":
        graph = adjacency
    elif test_format == "dense":
        graph = adjacency_matrix
    elif test_format == "networkx":
        import networkx
        graph = networkx.DiGraph(adjacency_matrix)
    elif test_format == "igraph":
        import igraph
        graph = igraph.Graph.Weighted_Adjacency(adjacency_matrix.tolist())

    if not InteractiveClass:
        return draw(graph, node_labels=node_labels, edge_labels=edge_labels, **kwargs)
    else:
        return InteractiveClass(graph, node_labels=node_labels, edge_labels=edge_labels, **kwargs)


if __name__ == "__main__":

    # create a figure for each possible combination of inputs

    arguments = OrderedDict(directed=(True, False),
                            strictly_positive=(True, False),
                            weighted=(True, False),
                            show_node_labels=(True,False),
                            show_edge_labels=(True, False),
                            test_format=('sparse', 'dense', 'networkx', 'igraph'),
                            InteractiveGraph=(None, InteractiveGraph))

    combinations = itertools.product(*arguments.values())

    for ii, combination in enumerate(combinations):
        print(ii, zip(arguments.keys(), combination))
        fig, ax = plt.subplots(1, 1, figsize=(16,16))
        kwargs = dict(zip(arguments.keys(), combination))
        graph = test(ax=ax, **kwargs)
        title = ''.join(['{}: {}, '.format(key, value) for (key, value) in kwargs.items()])
        filename = ''.join(['{}-{}_'.format(key, value) for (key, value) in kwargs.items()])
        filename = filename[:-1] # remove trailing underscore
        ax.set_title(title)
        fig.savefig('../figures/{}.pdf'.format(filename))
        plt.close()
