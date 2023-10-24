#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Implements the BaseMultiGraph and MultiGraph classes.
"""

import matplotlib.pyplot as plt

from ._graph_classes import BaseGraph, Graph
from ._parser import parse_multigraph, _parse_multigraph_edge_list
from ._artists import EdgeArtist
from ._edge_layout import (
    MultiGraphStraightEdgeLayout,
    MultiGraphCurvedEdgeLayout,
    MultiGraphBundledEdgeLayout,
    MultiGraphArcDiagramEdgeLayout,
)
from ._utils import (
    _simplify_multigraph,
)


class BaseMultiGraph(BaseGraph):
    """The MultiGraph base class.

    Parameters
    ----------
    edges : list
        The edges of the graph, with each edge being represented by a (source node ID, target node ID, edge key) tuple.
    nodes : list or None, default None
        List of nodes. Required argument if any node in the graph is unconnected.
        If None, `nodes` is initialised to the set of the flattened `edges`.
    node_layout : str or dict, default 'spring'
        If a string, the node positions are computed using the indicated method:

        - 'random'       : place nodes in random positions;
        - 'circular'     : place nodes regularly spaced on a circle;
        - 'spring'       : place nodes using a force-directed layout (Fruchterman-Reingold algorithm);
        - 'dot'          : place nodes using the Sugiyama algorithm; the graph should be directed and acyclic;
        - 'radial'       : place nodes radially using the Sugiyama algorithm; the graph should be directed and acyclic;
        - 'community'    : place nodes such that nodes belonging to the same community are grouped together;
        - 'bipartite'    : place nodes regularly spaced on two parallel lines;
        - 'multipartite' : place nodes regularly spaced on several parallel lines;
        - 'shell'        : place nodes regularly spaced on concentric circles;
        - 'geometric'    : place nodes according to the length of the edges between them.

        If a dict, keys are nodes and values are (x, y) positions.
    node_layout_kwargs : dict or None, default None
        Keyword arguments passed to node layout functions.
        See the documentation of the following functions for a full description of available options:

        - :py:func:`get_random_layout`
        - :py:func:`get_circular_layout`
        - :py:func:`get_fruchterman_reingold_layout`
        - :py:func:`get_sugiyama_layout`
        - :py:func:`get_radial_tree_layout`
        - :py:func:`get_community_layout`
        - :py:func:`get_bipartite_layout`
        - :py:func:`get_multipartite_layout`
        - :py:func:`get_shell_layout`
        - :py:func:`get_geometric_layout`

    node_shape : str or dict, default 'o'
        Node shape.
        If the type is str, all nodes have the same shape.
        If the type is dict, maps each node to an individual string representing the shape.
        The string specification is as for matplotlib.scatter marker, i.e. one of 'so^>v<dph8'.
    node_size : float or dict, default 3.
        Node size (radius).
        If the type is float, all nodes will have the same size.
        If the type is dict, maps each node to an individual size.

        .. note:: Values are rescaled by :py:const:`BASE_SCALE` (0.01) to be compatible with layout routines in igraph and networkx.

    node_edge_width : float or dict, default 0.5
        Line width of node marker border.
        If the type is float, all nodes have the same line width.
        If the type is dict, maps each node to an individual line width.

        .. note:: Values are rescaled by :py:const:`BASE_SCALE` (0.01) to be compatible with layout routines in igraph and networkx.

    node_color : matplotlib color specification or dict, default 'w'
        Node color.
        If the type is a string or RGBA array, all nodes have the same color.
        If the type is dict, maps each node to an individual color.
    node_edge_color : matplotlib color specification or dict, default :py:const:`DEFAULT_COLOR`
        Node edge color.
        If the type is a string or RGBA array, all nodes have the same edge color.
        If the type is dict, maps each node to an individual edge color.
    node_alpha : scalar or dict, default 1.
        Node transparency.
        If the type is a float, all nodes have the same transparency.
        If the type is dict, maps each node to an individual transparency.
    node_zorder : int or dict, default 2
        Order in which to plot the nodes.
        If the type is an int, all nodes have the same zorder.
        If the type is dict, maps each node to an individual zorder.
    node_labels : bool or dict, (default False)
        If False, the nodes are unlabelled.
        If True, the nodes are labelled with their node IDs.
        If the node labels are to be distinct from the node IDs, supply a dictionary mapping nodes to node labels.
        Only nodes in the dictionary are labelled.
    node_label_offset: float or tuple, default (0., 0.)
        A (dx, dy) tuple specifies the exact offset from the node position.
        If a single scalar delta is specified, the value is interpreted as a distance,
        and the label is placed delta away from the node position while trying to
        reduce node/label, node/edge, and label/label overlaps.
    node_label_fontdict : dict
        Keyword arguments passed to matplotlib.text.Text.
        For a full list of available arguments see the matplotlib documentation.
        The following default values differ from the defaults for matplotlib.text.Text:

        - :code:`size` (adjusted to fit into node artists if offset is (0, 0))
        - :code:`horizontalalignment` (default here: :code:`'center'`)
        - :code:`verticalalignment` (default here: :code:`'center'`)
        - :code:`clip_on` (default here: :code:`False`)
        - :code:`zorder` (default here: :code:`inf`)

    edge_width : float or dict, default 1.
        Width of edges.
        If the type is a float, all edges have the same width.
        If the type is dict, maps each edge to an individual width.

        .. note:: Value is rescaled by :py:const:`BASE_SCALE` (0.01) to be compatible with layout routines in igraph and networkx.

    edge_color : matplotlib color specification or dict, default :py:const:`DEFAULT_COLOR`
        Edge color.
        If the type is a string or RGBA array, all edges have the same color.
        If the type is dict, maps each edge to an individual color.
    edge_alpha : float or dict, default 1.
        The edge transparency,
        If the type is a float, all edges have the same transparency.
        If the type is dict, maps each edge to an individual transparency.
    edge_zorder : int or dict, default 1
        Order in which to plot the edges.
        If the type is an int, all edges have the same zorder.
        If the type is dict, maps each edge to an individual zorder.
        If None, the edges will be plotted in the order they appear in the 'graph' argument.
        Hint: graphs typically appear more visually pleasing if darker edges are plotted on top of lighter edges.
    arrows : bool, default False
        If True, draw edges with arrow heads.
    edge_layout : str or dict (default 'straight')
        If edge_layout is a string, determine the layout internally:

        - 'straight' : draw edges as straight lines
        - 'curved'   : draw edges as curved splines; the spline control points are optimised to avoid other nodes and edges
        - 'bundled'  : draw edges as edge bundles
        - 'arc'      : draw edges as arcs with a fixed curvature

        If edge_layout is a dict, the keys are edges and the values are edge paths
        in the form iterables of (x, y) tuples, the edge segments.
    edge_layout_kwargs : dict, default None
        Keyword arguments passed to edge layout functions.
        See the documentation of the following functions for a full description of available options:

        - :py:func:`get_straight_edge_paths`
        - :py:func:`get_curved_edge_paths`
        - :py:func:`get_bundled_edge_paths`
        - :py:func:`get_arced_edge_paths`

    edge_labels : bool or dict, default False
        If False, the edges are unlabelled.
        If True, the edges are labelled with their edge IDs.
        If the edge labels are to be distinct from the edge IDs, supply a dictionary mapping edges to edge labels.
        Only edges in the dictionary are labelled.
    edge_label_position : float, default 0.5
        Relative position along the edge where the label is placed.

        - head   : 0.
        - centre : 0.5
        - tail   : 1.

    edge_label_rotate : bool, default True
        If True, edge labels are rotated such that they have the same orientation as their edge.
        If False, edge labels are not rotated; the angle of the text is parallel to the axis.
    edge_label_fontdict : dict
        Keyword arguments passed to matplotlib.text.Text.
        For a full list of available arguments see the matplotlib documentation.
        The following default values differ from the defaults for matplotlib.text.Text:

        - :code:`horizontalalignment` (default here: :code:`'center'`),
        - :code:`verticalalignment` (default here: :code:`'center'`)
        - :code:`clip_on` (default here: code:`False`),
        - :code:`bbox` (default here: :code:`dict(color='white', pad=0)`,
        - :code:`zorder` (default here: :code:`inf`),
        - :code:`rotation` (determined by :code:`edge_label_rotate` argument)

    origin : tuple, default (0., 0.)
        The lower left hand corner of the bounding box specifying the extent of the canvas.
    scale : tuple, default (1., 1.)
        The width and height of the bounding box specifying the extent of the canvas.
    prettify : bool, default True
        If True, despine and remove ticks and tick labels.
        Set figure background to white. Set axis aspect to equal.
    ax : matplotlib.axis instance or None, default None
        Axis to plot onto; if none specified, one will be instantiated with plt.gca().

    Attributes
    ----------
    node_artists : dict
        Mapping of node IDs to matplotlib PathPatch artists.
    edge_artists : dict
        Mapping of edge IDs to matplotlib PathPatch artists.
    node_label_artists : dict
        Mapping of node IDs to matplotlib text objects (if applicable).
    edge_label_artists : dict
        Mapping of edge IDs to matplotlib text objects (if applicable).
    node_positions : dict node : (x, y) tuple
        Mapping of node IDs to node positions.

    See also
    --------
    :py:class:`BaseGraph`, :py:class:`MultiGraph`

    """

    def _initialize_edges(self, edges):
        return _parse_multigraph_edge_list(edges)


    def _get_node_positions(self, node_layout, node_layout_kwargs, edges, *args, **kwargs):
        return super()._get_node_positions(node_layout, node_layout_kwargs, _simplify_multigraph(edges), *args, **kwargs)


    def _initialize_edge_layout(self, edge_layout, edge_layout_kwargs):
        if edge_layout_kwargs is None:
            edge_layout_kwargs = dict()

        # TODO: improve handling of dict arguments such as selfloop_radius
        selfloops = [(source, target) for (source, target, eid) in self.edges if source==target]
        if selfloops:
            if 'selfloop_radius' in edge_layout_kwargs:
                edge_layout_kwargs['selfloop_radius'] = \
                    _normalize_numeric_argument(edge_layout_kwargs['selfloop_radius'], selfloops, 'selfloop_radius')
            else:
                edge_layout_kwargs['selfloop_radius'] = \
                    {(node, node) : 1.5 * self.node_size[node] for node, _ in selfloops}

        if isinstance(edge_layout, str):
            if edge_layout == "straight":
                edge_layout = MultiGraphStraightEdgeLayout(self.edges, self.node_positions, self.edge_width, **edge_layout_kwargs)
            elif edge_layout == "curved":
                edge_layout_kwargs.setdefault('node_size', self.node_size)
                edge_layout_kwargs.setdefault('origin', self.origin)
                edge_layout_kwargs.setdefault('scale', self.scale)
                edge_layout = MultiGraphCurvedEdgeLayout(self.edges, self.node_positions, self.edge_width, **edge_layout_kwargs)
            elif edge_layout == "bundled":
                edge_layout = MultiGraphBundledEdgeLayout(self.edges, self.node_positions, self.edge_width, **edge_layout_kwargs)
            elif edge_layout == "arc":
                edge_layout = MultiGraphArcDiagramEdgeLayout(self.edges, self.node_positions, self.edge_width, **edge_layout_kwargs)
            else:
                raise NotImplementedError(f"Variable edge_layout one of 'straight', 'curved', 'bundled', or 'arc', not {edge_layout}")
            edge_paths = edge_layout.get()

        elif isinstance(edge_layout, dict):
            _check_completeness(edge_layout, self.edges, 'edge_layout')
            edge_paths = edge_layout
            edge_layout = MultigraphStraightEdgeLayout(self.edges, self.node_positions, self.edge_width, **edge_layout_kwargs)
            edge_layout.edge_paths.update(edge_paths)

        else:
            raise TypeError("Variable `edge_layout` either a string or a dict mapping edges to edge paths.")

        return edge_layout


    def draw_edges(self, edges, edge_path, edge_width, edge_color, edge_alpha,
                   edge_zorder, arrows, node_artists):
        """Draw or update edge artists.

        Parameters
        ----------
        edges : list
            The edges of the graph, with each edge being represented by a
            (source node ID, target node ID, edge ID) tuple.
        edge_path : dict
            Mapping of edges to arrays of (x, y) tuples, the edge path coordinates.
        edge_width : dict
            Mapping of edges to floats, the edge widths.
        edge_color : dict
            Mapping of edges to valid matplotlib color specifications, the edge colors.
        edge_alpha : dict
            Mapping of edges to floats, the edge transparencies.
        edge_zorder : dict
            Mapping of edges to ints, the edge z-order values.
        arrows : bool
            If True, draw edges with arrow heads.
        node_artists : dict
            Mapping of nodes to node artists. Required to offset edges from nodes.

        Returns
        -------
        self.edge_artists: dict
            Updates mapping of edges to corresponding edge artists.

        """

        for edge in edges:

            curved = False if (len(edge_path[edge]) == 2) else True

            source, target, eid = edge

            if arrows:
                head_length = 2 * edge_width[edge]
                head_width = 3 * edge_width[edge]
            else:
                head_length = 0
                head_width = 0

            edge_artist = EdgeArtist(
                midline     = edge_path[edge],
                width       = edge_width[edge],
                facecolor   = edge_color[edge],
                alpha       = edge_alpha[edge],
                head_length = head_length,
                head_width  = head_width,
                edgecolor   = 'none',
                linewidth   = 0.,
                head_offset = node_artists[target].get_head_offset(edge_path[edge]),
                tail_offset = node_artists[source].get_tail_offset(edge_path[edge]),
                shape       = "full",
                curved      = curved,
                zorder      = edge_zorder[edge],
            )
            self.ax.add_patch(edge_artist)

            if edge in self.edge_artists:
                self.edge_artists[edge].remove()
            self.edge_artists[edge] = edge_artist


class MultiGraph(BaseMultiGraph, Graph):
    """Parses the given graph data object and initialises the :py:class:`BaseMultiGraph` object.

    If the given graph includes edge weights, then these are mapped to colors using the `edge_cmap` parameter.

    Parameters
    ----------
    graph: various formats
        Graph object to plot. Various input formats are supported.
        In order of precedence:

        - Edge list:
          Iterable of (source node ID, target node ID, edge key) or
          (source node ID, target node ID, edge key, weight) tuples,
          or equivalent (E, 3) or (E, 4) ndarray (where E is the number of edges).
        - Adjacency matrix:
          A (V, V, L) ndarray, where V is the number of nodes/vertices, and L is the number of layers.
          The absence of a connection is indicated by a zero.
        - networkx.MultiGraph or igraph.Graph object

    node_layout : str or dict, default 'spring'
        If a string, the node positions are computed using the indicated method:

        - 'random'       : place nodes in random positions;
        - 'circular'     : place nodes regularly spaced on a circle;
        - 'spring'       : place nodes using a force-directed layout (Fruchterman-Reingold algorithm);
        - 'dot'          : place nodes using the Sugiyama algorithm; the graph should be directed and acyclic;
        - 'radial'       : place nodes radially using the Sugiyama algorithm; the graph should be directed and acyclic;
        - 'community'    : place nodes such that nodes belonging to the same community are grouped together;
        - 'bipartite'    : place nodes regularly spaced on two parallel lines;
        - 'multipartite' : place nodes regularly spaced on several parallel lines;
        - 'shell'        : place nodes regularly spaced on concentric circles;
        - 'geometric'    : place nodes according to the length of the edges between them.

        If a dict, keys are nodes and values are (x, y) positions.
    node_layout_kwargs : dict or None, default None
        Keyword arguments passed to node layout functions.
        See the documentation of the following functions for a full description of available options:

        - :py:func:`get_random_layout`
        - :py:func:`get_circular_layout`
        - :py:func:`get_fruchterman_reingold_layout`
        - :py:func:`get_sugiyama_layout`
        - :py:func:`get_radial_tree_layout`
        - :py:func:`get_community_layout`
        - :py:func:`get_bipartite_layout`
        - :py:func:`get_multipartite_layout`
        - :py:func:`get_shell_layout`
        - :py:func:`get_geometric_layout`

    node_shape : str or dict, default 'o'
        Node shape.
        If the type is str, all nodes have the same shape.
        If the type is dict, maps each node to an individual string representing the shape.
        The string specification is as for matplotlib.scatter marker, i.e. one of 'so^>v<dph8'.
    node_size : float or dict, default 3.
        Node size (radius).
        If the type is float, all nodes will have the same size.
        If the type is dict, maps each node to an individual size.

        .. note:: Values are rescaled by :py:const:`BASE_SCALE` (0.01) to be compatible with layout routines in igraph and networkx.

    node_edge_width : float or dict, default 0.5
        Line width of node marker border.
        If the type is float, all nodes have the same line width.
        If the type is dict, maps each node to an individual line width.

        .. note:: Values are rescaled by :py:const:`BASE_SCALE` (0.01) to be compatible with layout routines in igraph and networkx.

    node_color : matplotlib color specification or dict, default 'w'
        Node color.
        If the type is a string or RGBA array, all nodes have the same color.
        If the type is dict, maps each node to an individual color.
    node_edge_color : matplotlib color specification or dict, default :py:const:`DEFAULT_COLOR`
        Node edge color.
        If the type is a string or RGBA array, all nodes have the same edge color.
        If the type is dict, maps each node to an individual edge color.
    node_alpha : scalar or dict, default 1.
        Node transparency.
        If the type is a float, all nodes have the same transparency.
        If the type is dict, maps each node to an individual transparency.
    node_zorder : int or dict, default 2
        Order in which to plot the nodes.
        If the type is an int, all nodes have the same zorder.
        If the type is dict, maps each node to an individual zorder.
    node_labels : bool or dict, (default False)
        If False, the nodes are unlabelled.
        If True, the nodes are labelled with their node IDs.
        If the node labels are to be distinct from the node IDs, supply a dictionary mapping nodes to node labels.
        Only nodes in the dictionary are labelled.
    node_label_offset: float or tuple, default (0., 0.)
        A (dx, dy) tuple specifies the exact offset from the node position.
        If a single scalar delta is specified, the value is interpreted as a distance,
        and the label is placed delta away from the node position while trying to
        reduce node/label, node/edge, and label/label overlaps.
    node_label_fontdict : dict
        Keyword arguments passed to matplotlib.text.Text.
        For a full list of available arguments see the matplotlib documentation.
        The following default values differ from the defaults for matplotlib.text.Text:

        - :code:`size` (adjusted to fit into node artists if offset is (0, 0))
        - :code:`horizontalalignment` (default here: :code:`'center'`)
        - :code:`verticalalignment` (default here: :code:`'center'`)
        - :code:`clip_on` (default here: :code:`False`)
        - :code:`zorder` (default here: :code:`inf`)

    edge_width : float or dict, default 1.
        Width of edges.
        If the type is a float, all edges have the same width.
        If the type is dict, maps each edge to an individual width.

        .. note:: Value is rescaled by :py:const:`BASE_SCALE` (0.01) to be compatible with layout routines in igraph and networkx.

    edge_color : matplotlib color specification or dict, default :py:const:`DEFAULT_COLOR`
        Edge color.
        If the type is a string or RGBA array, all edges have the same color.
        If the type is dict, maps each edge to an individual color.
    edge_alpha : float or dict, default 1.
        The edge transparency,
        If the type is a float, all edges have the same transparency.
        If the type is dict, maps each edge to an individual transparency.
    edge_zorder : int or dict, default 1
        Order in which to plot the edges.
        If the type is an int, all edges have the same zorder.
        If the type is dict, maps each edge to an individual zorder.
        If None, the edges will be plotted in the order they appear in the 'graph' argument.
        Hint: graphs typically appear more visually pleasing if darker edges are plotted on top of lighter edges.
    arrows : bool, default False
        If True, draw edges with arrow heads.
    edge_layout : str or dict (default 'straight')
        If edge_layout is a string, determine the layout internally:

        - 'straight' : draw edges as straight lines
        - 'curved'   : draw edges as curved splines; the spline control points are optimised to avoid other nodes and edges
        - 'bundled'  : draw edges as edge bundles
        - 'arc'      : draw edges as arcs with a fixed curvature

        If edge_layout is a dict, the keys are edges and the values are edge paths
        in the form iterables of (x, y) tuples, the edge segments.
    edge_layout_kwargs : dict, default None
        Keyword arguments passed to edge layout functions.
        See the documentation of the following functions for a full description of available options:

        - :py:func:`get_straight_edge_paths`
        - :py:func:`get_curved_edge_paths`
        - :py:func:`get_bundled_edge_paths`
        - :py:func:`get_arced_edge_paths`

    edge_labels : bool or dict, default False
        If False, the edges are unlabelled.
        If True, the edges are labelled with their edge IDs.
        If the edge labels are to be distinct from the edge IDs, supply a dictionary mapping edges to edge labels.
        Only edges in the dictionary are labelled.
    edge_label_position : float, default 0.5
        Relative position along the edge where the label is placed.

        - head   : 0.
        - centre : 0.5
        - tail   : 1.

    edge_label_rotate : bool, default True
        If True, edge labels are rotated such that they have the same orientation as their edge.
        If False, edge labels are not rotated; the angle of the text is parallel to the axis.
    edge_label_fontdict : dict
        Keyword arguments passed to matplotlib.text.Text.
        For a full list of available arguments see the matplotlib documentation.
        The following default values differ from the defaults for matplotlib.text.Text:

        - :code:`horizontalalignment` (default here: :code:`'center'`),
        - :code:`verticalalignment` (default here: :code:`'center'`)
        - :code:`clip_on` (default here: code:`False`),
        - :code:`bbox` (default here: :code:`dict(color='white', pad=0)`,
        - :code:`zorder` (default here: :code:`inf`),
        - :code:`rotation` (determined by :code:`edge_label_rotate` argument)

    origin : tuple, default (0., 0.)
        The lower left hand corner of the bounding box specifying the extent of the canvas.
    scale : tuple, default (1., 1.)
        The width and height of the bounding box specifying the extent of the canvas.
    prettify : bool, default True
        If True, despine and remove ticks and tick labels.
        Set figure background to white. Set axis aspect to equal.
    ax : matplotlib.axis instance or None, default None
        Axis to plot onto; if none specified, one will be instantiated with plt.gca().

    Attributes
    ----------
    node_artists : dict
        Mapping of node IDs to matplotlib PathPatch artists.
    edge_artists : dict
        Mapping of edge IDs to matplotlib PathPatch artists.
    node_label_artists : dict
        Mapping of node IDs to matplotlib text objects (if applicable).
    edge_label_artists : dict
        Mapping of edge IDs to matplotlib text objects (if applicable).
    node_positions : dict node : (x, y) tuple
        Mapping of node IDs to node positions.

    See also
    --------
    :py:class:`BaseMultiGraph`, :py:class:`Graph`, :py:class:`InteractiveMultiGraph`

    """

    def __init__(self, *args, **kwargs):
        Graph.__init__(self, *args, **kwargs)


    def _parse_input(self, graph):
        return parse_multigraph(graph)
