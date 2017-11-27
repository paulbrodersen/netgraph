#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from scipy.sparse import coo_matrix, spdiags
from collections import OrderedDict

BASE_NODE_SIZE = 1e-2 # i.e. node sizes are in percent of axes space (x,y <- [0, 1], [0,1])
BASE_EDGE_WIDTH = 1e-2 # i.e. edge widths are in percent of axis space (x,y <- [0, 1], [0,1])


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
    edge_list, edge_weight = parse_graph(graph)

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
    if _is_directed(edge_list):
        kwargs.setdefault('draw_arrows', True)
    else:
        kwargs.setdefault('draw_arrows', False)

    # Initialise node positions if none are given.
    if node_positions is None:
        node_positions = fruchterman_reingold_layout(edge_list)

    # Create axis if none is given.
    if ax is None:
        ax = plt.gca()

    # Draw plot elements.
    draw_edges(edge_list, node_positions, ax=ax, **kwargs)
    draw_nodes(node_positions, ax=ax, **kwargs)

    if node_labels is not None:
        draw_node_labels(node_labels, node_positions, ax=ax, **kwargs)

    if edge_labels is not None:
        draw_edge_labels(edge_labels, node_positions, ax=ax, **kwargs)

    # Improve default layout of axis.
    if 'node_size' in kwargs:
        _update_view(node_positions, ax=ax, node_size=kwargs['node_size'])
    else:
        _update_view(node_positions, ax=ax)
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
        return _parse_edge_list(adjacency), None
    elif columns == 3:
        edge_list = _parse_edge_list(adjacency[:,:2])
        edge_weights = {(source, target) : weight for (source, target, weight) in adjacency}

        if len(set(edge_weights.values())) > 1:
            return edge_list, edge_weights
        else:
            return edge_list, None
    else:
        raise ValueError("Graph specification in sparse matrix format needs to consist of an iterable of tuples of length 2 or 3. Got iterable of tuples of length {}.".format(columns))


def _parse_adjacency_matrix(adjacency):
    sources, targets = np.where(adjacency)
    edge_list = list(zip(sources.tolist(), targets.tolist()))
    edge_weights = {(source, target): adjacency[source, target] for (source, target) in edge_list}
    if len(set(list(edge_weights.values()))) == 1:
        return edge_list, None
    return edge_list, edge_weights


def _parse_networkx_graph(graph, attribute_name='weight'):
    edge_list = list(graph.edges())
    try:
        edge_weights = {edge : graph.get_edge_data(*edge)[attribute_name] for edge in edge_list}
    except KeyError: # no weights
        edge_weights = None
    return edge_list, edge_weights


def _parse_igraph_graph(graph):
    edge_list = [(edge.source, edge.target) for edge in graph.es()]
    if graph.is_weighted():
        edge_weights = {(edge.source, edge.target) : edge['weight'] for edge in graph.es()}
    else:
        edge_weights = None
    return edge_list, edge_weights


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
    values = np.array(mydict.values())

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
    zorder = np.argsort(np.sum(color_dict.values(), axis=1)) # assumes RGB specification
    zorder = np.max(zorder) - zorder # reverse order as greater values correspond to lighter colors
    zorder = {key: index for key, index in zip(color_dict.keys(), zorder)}
    return zorder


def _is_directed(edge_list):
    # test for bi-directional edges
    for (source, target) in edge_list:
        if ((target, source) in edge_list) and (source != target):
            return True
    return False


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
        Mapping of nodes to (x, y) positions

    node_shape : string or dict key : string (default 'o')
       The shape of the node. Specification is as for matplotlib.scatter
       marker, i.e. one of 'so^>v<dph8'.
       If a single string is provided all nodes will have the same shape.

    node_size : scalar or dict node : float (default 3.)
       Size (radius) of nodes in percent of axes space.

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
       Size (radius) of nodes in percent of axes space.

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
    edge_list: m-long iterable of 2-tuples or equivalent (such as (m, 2) ndarray)
        List of edges. Each tuple corresponds to an edge defined by (source, target).

    node_positions : dict key : (float, float)
        Mapping of nodes to (x,y) positions

    node_size : scalar or (n,) or dict key : float (default 3.)
        Size (radius) of nodes in percent of axes space.
        Used to offset edges when drawing arrow heads,
        such that the arrow heads are not occluded.
        If draw_nodes() and draw_edges() are called independently,
        make sure to set this variable to the same value.

    edge_width : float or dict (source, key) : width (default 1.)
        Line width of edges.

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
                [0.0, 0.0],                  # tip
                [-hl, -hw / 2.0],             # leftmost
                [-hl * (1 - hs), -lw / 2.0],  # meets stem
                [-length, -lw / 2.0],          # bottom left
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
                    raise ValueError("Got unknown shape: %s" % shape)
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

    artists = dict()  # there is no text collection so we'll fake one
    for node, label in node_labels.items():
        x, y = node_positions[node]
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


def draw_edge_labels(edge_labels,
                     node_positions,
                     edge_label_font_size=10,
                     edge_label_font_color='k',
                     edge_label_font_family='sans-serif',
                     edge_label_font_weight='normal',
                     edge_label_font_alpha=1.,
                     edge_label_bbox=None,
                     edge_label_horizontalalignment='center',
                     edge_label_verticalalignment='center',
                     clip_on=False,
                     ax=None,
                     rotate=True,
                     edge_label_zorder=10000,
                     **kwargs):
    """
    Draw edge labels.

    Arguments
    ---------

    node_positions : dict node : (float, float)
        Mapping of nodes to (x, y) positions

    edge_labels : dict (source, target) : str
        Mapping of edges to edge labels.
        Only edges in the dictionary are labelled.

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

    ax : matplotlib.axis instance or None (default None)
       Axis to plot onto; if none specified, one will be instantiated with plt.gca().

    Returns
    -------
    artists: dict (source, target) : text object
        Mapping of edges to edge label artists.

    @reference
    Borrowed with minor modifications from networkx/drawing/nx_pylab.py

    """

    # draw labels centered on the midway point of the edge
    label_pos = 0.5

    if ax is None:
        ax = plt.gca()

    text_items = {}
    for (n1, n2), label in edge_labels.items():

        if n1 != n2:

            (x1, y1) = node_positions[n1]
            (x2, y2) = node_positions[n2]
            (x, y) = (x1 * label_pos + x2 * (1.0 - label_pos),
                      y1 * label_pos + y2 * (1.0 - label_pos))

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
        maxs = np.max(node_size.values()) * BASE_NODE_SIZE
    else:
        maxs = node_size * BASE_NODE_SIZE

    maxx, maxy = np.max(node_positions.values(), axis=0)
    minx, miny = np.min(node_positions.values(), axis=0)

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
                                dim=2):
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

    nodes = np.unique(edge_list)
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

    A = _edge_list_to_sparse_matrix(edge_list, edge_weights)

    if total_nodes > 500:  # sparse solver for large graphs
        pos = _sparse_fruchterman_reingold(A, k, pos_arr, fixed, iterations, dim)
    else:
        pos = _dense_fruchterman_reingold(A.toarray(), k, pos_arr, fixed, iterations, dim)

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

    if pos is None:
        # random initial positions
        pos = np.asarray(np.random.random((nnodes, dim)), dtype=A.dtype)
    else:
        # make sure positions are of same type as matrix
        pos = pos.astype(A.dtype)

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


def _sparse_fruchterman_reingold(A, k=None, pos=None, fixed=None,
                                 iterations=50, dim=2):
    # Position nodes in adjacency matrix A using Fruchterman-Reingold
    # Entry point for NetworkX graph is fruchterman_reingold_layout()
    # Sparse version

    nnodes, _ = A.shape

    # make sure we have a list of lists representation
    try:
        A = A.tolil()
    except:
        A = (coo_matrix(A)).tolil()

    if pos is None:
        # random initial positions
        pos = np.asarray(np.random.random((nnodes, dim)), dtype=A.dtype)
    else:
        # make sure positions are of same type as matrix
        pos = pos.astype(A.dtype)

    # no fixed nodes
    if fixed is None:
        fixed = []

    # optimal distance between nodes
    if k is None:
        k = np.sqrt(1.0/nnodes)
    # the initial "temperature"  is about .1 of domain area (=1x1)
    # this is the largest step allowed in the dynamics.
    t = 0.1
    # simple cooling scheme.
    # linearly step down by dt on each iteration so last iteration is size dt.
    dt = t / float(iterations+1)

    displacement = np.zeros((dim, nnodes))
    for iteration in range(iterations):
        displacement *= 0
        # loop over rows
        for i in range(A.shape[0]):
            if i in fixed:
                continue
            # difference between this row's node position and all others
            delta = (pos[i] - pos).T
            # distance between points
            distance = np.sqrt((delta**2).sum(axis=0))
            # enforce minimum distance of 0.01
            distance = np.where(distance < 0.01, 0.01, distance)
            # the adjacency matrix row
            Ai = np.asarray(A.getrowview(i).toarray())
            # displacement "force"
            displacement[:, i] +=\
                (delta * (k * k / distance**2 - Ai * distance / k)).sum(axis=1)
        # update positions
        length = np.sqrt((displacement**2).sum(axis=0))
        length = np.where(length < 0.01, 0.1, length)
        pos += (displacement * t / length).T
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


def _edge_list_to_sparse_matrix(edge_list, edge_weights=None):

    nodes = np.unique(edge_list)
    node_to_idx = dict(zip(nodes, range(len(nodes))))
    sources = [node_to_idx[source] for source, _ in edge_list]
    targets = [node_to_idx[target] for _, target in edge_list]

    total_nodes = len(nodes)
    shape = (total_nodes, total_nodes)

    if edge_weights:
        weights = [edge_weights[edge] for edge in edge_list]
        arr = coo_matrix((weights, (sources, targets)), shape=shape)
    else:
        arr = coo_matrix((np.ones((len(sources))), (sources, targets)), shape=shape)

    return arr


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
        self.edge_list, self.edge_weight = parse_graph(graph)

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
        if _is_directed(self.edge_list):
            kwargs.setdefault('draw_arrows', True)
        else:
            kwargs.setdefault('draw_arrows', False)

        # Initialise node positions if none are given or already present.
        if node_positions:
            if not hasattr(self, 'node_positions'):
                self.node_positions = node_positions
            else:
                self.node_positions.update(node_positions)
        else:
            if not hasattr(self, 'node_positions'):
                self.node_positions = fruchterman_reingold_layout(self.edge_list)

        # Draw plot elements.
        self.draw_edges(self.edge_list, self.node_positions, ax=self.ax, **kwargs)
        self.draw_nodes(self.node_positions, ax=self.ax, **kwargs)

        if node_labels:
            if not hasattr(self, 'node_labels'):
                self.node_labels = node_labels
            else:
                self.node_labels.update(node_labels)
            self.draw_node_labels(self.node_labels, self.node_positions, ax=self.ax, **kwargs)

        if edge_labels:
            if not hasattr(self, 'edge_labels'):
                self.edge_labels = edge_labels
            else:
                self.edge_labels.update(edge_labels)
            self.draw_edge_labels(self.edge_labels, self.node_positions, ax=self.ax, **kwargs)

        # Improve default layout of axis.
        self._update_view()

        _make_pretty(self.ax)


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
           Size (radius) of nodes in percent of axes space.

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
            self.node_face_artists = dict()
        if not hasattr(self, 'node_edge_artists'):
            self.node_edge_artists = dict()

        self.node_face_artists.update(node_faces)
        self.node_edge_artists.update(node_edges)


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
            Size (radius) of nodes in percent of axes space.
            Used to offset edges when drawing arrow heads,
            such that the arrow heads are not occluded.
            If draw_nodes() and draw_edges() are called independently,
            make sure to set this variable to the same value.

        edge_width : float or dict (source, key) : width (default 1.)
            Line width of edges.

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
            self.edge_artists = dict()

        self.edge_artists.update(artists)


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
            self.node_label_artists = dict()

        self.node_label_artists.update(artists)


    def draw_edge_labels(self, *args, **kwargs):
        """
        Draw edge labels.

        Arguments
        ---------

        node_positions : dict node : (float, float)
            Mapping of nodes to (x, y) positions

        edge_labels : dict (source, target) : str
            Mapping of edges to edge labels.
            Only edges in the dictionary are labelled.

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
            self.edge_label_artists = dict()

        self.edge_label_artists.update(artists)


    def _update_view(self):
        # Pad x and y limits as patches are not registered properly
        # when matplotlib sets axis limits automatically.
        # Hence we need to set them manually.

        max_edge_radius = np.max([artist.radius for artist in self.node_edge_artists.values()])
        max_face_radius = np.max([artist.radius for artist in self.node_face_artists.values()])
        max_radius = np.max([max_edge_radius, max_face_radius])

        maxx, maxy = np.max(self.node_positions.values(), axis=0)
        minx, miny = np.min(self.node_positions.values(), axis=0)

        w = maxx-minx
        h = maxy-miny
        padx, pady = 0.05*w + max_radius, 0.05*h + max_radius
        corners = (minx-padx, miny-pady), (maxx+padx, maxy+pady)

        self.ax.update_datalim(corners)
        self.ax.autoscale_view()
        self.ax.get_figure().canvas.draw()


class InteractiveGraph(Graph):
    """
    Notes:
    ------
    Methods adapted with some modifications from:
    https://stackoverflow.com/questions/47293499/window-select-multiple-artists-and-drag-them-on-canvas/47312637#47312637
    """

    def __init__(self, *args, **kwargs):
        # Initialise as before.
        Graph.__init__(self, *args, **kwargs)

        self._alpha = {key: artist.get_alpha() for key, artist in self.node_face_artists.items()}

        self.fig = self.ax.get_figure()
        self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)

        self._currently_selecting = False
        self._currently_dragging = False
        self._selected_artists = {}
        self._offset = {}
        self._rect = plt.Rectangle((0,0),1,1, linestyle="--",
                                  edgecolor="crimson", fill=False)
        self.ax.add_patch(self._rect)
        self._rect.set_visible(False)

        self._x0 = 0
        self._y0 = 0
        self._x1 = 0
        self._y1 = 0


    def _on_press(self, event):

        # reset rectangle
        self._x0 = event.xdata
        self._y0 = event.ydata
        self._x1 = event.xdata
        self._y1 = event.ydata

        # is the press over some artist
        is_on_artist = False
        for key, artist in self.node_face_artists.items():
            if artist.contains(event)[0]:
                is_on_artist = True
                self._select_artist(key, artist)

        if is_on_artist:
            # add clicked artist to selection
            # start dragging
            self._currently_dragging = True
            self._offset = {key : artist.center - np.array([event.xdata, event.ydata]) for key, artist in self._selected_artists.items()}
        else:
            # start selecting
            self._deselect_artists()
            self._currently_selecting = True


    def _on_release(self, event):
        if self._currently_selecting:
            for key, artist in self.node_face_artists.items():
                if self._is_inside_rect(*artist.center):
                    self._select_artist(key, artist)
            self.fig.canvas.draw_idle()
            self._currently_selecting = False
            self._rect.set_visible(False)

        elif self._currently_dragging:
            self._currently_dragging = False


    def _on_motion(self, event):
        if event.inaxes:
            if self._currently_dragging:
                self._update_nodes(event)
                self._update_edges()
                self.fig.canvas.draw_idle()
            elif self._currently_selecting:
                self._x1 = event.xdata
                self._y1 = event.ydata
                # add rectangle for selection here
                self._selector_on()
                self.fig.canvas.draw_idle()


    def _is_inside_rect(self, x, y):
        xlim = np.sort([self._x0, self._x1])
        ylim = np.sort([self._y0, self._y1])
        if (xlim[0]<=x) and (x<xlim[1]) and (ylim[0]<=y) and (y<ylim[1]):
            return True
        else:
            return False


    def _select_artist(self, key, artist):
        if key not in self._selected_artists:
            alpha = artist.get_alpha()
            artist.set_alpha(0.5 * alpha)
            self._selected_artists[key] = artist


    def _deselect_artists(self):
        for key, artist in self._selected_artists.items():
            artist.set_alpha(self._alpha[key])
        self._selected_artists = {}


    def _selector_on(self):
        self._rect.set_visible(True)
        xlim = np.sort([self._x0, self._x1])
        ylim = np.sort([self._y0, self._y1])
        self._rect.set_xy((xlim[0],ylim[0] ) )
        self._rect.set_width(np.diff(xlim))
        self._rect.set_height(np.diff(ylim))


    def _update_nodes(self, event):
        for key in self._selected_artists.keys():
            pos = np.array([event.xdata, event.ydata]) + self._offset[key]
            self._move_node(key, pos)


    def _move_node(self, node, pos):
        self.node_positions[node] = pos
        self.node_edge_artists[node].center = pos
        self.node_face_artists[node].center = pos
        try:
            self.node_label_artists[node].set_position(pos)
        except AttributeError: # no node labels
            pass


    def _update_edges(self):
        # get edges that need to move
        edges = []
        for key in self._selected_artists.keys():
            edges.extend([edge for edge in self.edge_list if key in edge])
        edges = list(set(edges))

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


    def _update_edge_labels(self, edges, node_positions, rotate=True): # TODO: pass 'rotate' properly

        # draw labels centered on the midway point of the edge
        label_pos = 0.5

        for (n1, n2) in edges:
            (x1, y1) = node_positions[n1]
            (x2, y2) = node_positions[n2]
            (x, y) = (x1 * label_pos + x2 * (1.0 - label_pos),
                      y1 * label_pos + y2 * (1.0 - label_pos))

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


def test(n=20, p=0.15,
         directed=True,
         weighted=True,
         strictly_positive=False,
         test_format='sparse',
         show_node_labels=False,
         show_edge_labels=False,
         InteractiveClass=False,
         ax=None):

    adjacency_matrix = _get_random_weight_matrix(n, p,
                                                 directed=directed,
                                                 strictly_positive=strictly_positive,
                                                 weighted=weighted)

    sources, targets = np.where(adjacency_matrix)
    weights = adjacency_matrix[sources, targets]
    adjacency = np.c_[sources, targets, weights]

    if show_node_labels:
        node_labels = {node: str(int(node)) for node in np.unique(adjacency[:,:2])}
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
        graph = networkx.from_numpy_array(adjacency_matrix, networkx.DiGraph)
    elif test_format == "igraph":
        import igraph
        graph = igraph.Graph.Weighted_Adjacency(adjacency_matrix.tolist())

    if not InteractiveClass:
        return draw(graph, node_labels=node_labels, edge_labels=edge_labels, ax=ax)
    else:
        return InteractiveClass(graph, node_labels=node_labels, edge_labels=edge_labels, ax=ax)


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

    for combination in combinations:
        fig, ax = plt.subplots(1, 1, figsize=(16,16))
        kwargs = dict(zip(arguments.keys(), combination))
        graph = test(ax=ax, **kwargs)
        title = ''.join(['{}: {}, '.format(key, value) for (key, value) in kwargs.items()])
        filename = ''.join(['{}-{}_'.format(key, value) for (key, value) in kwargs.items()])
        filename = filename[:-1] # remove trailing underscore
        ax.set_title(title)
        fig.savefig('{}.pdf'.format(filename))
        plt.close()
