#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Implements the BaseGraph, Graph, and InteractiveGraph classes.
"""
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from uuid import uuid4

from ._utils import (
    _get_unique_nodes,
    _get_angle,
    _get_interior_angle_between,
    _get_orthogonal_unit_vector,
    _get_point_along_spline,
    _get_tangent_at_point,
    _make_pretty,
    _rank,
    _get_n_points_on_a_circle,
    _edge_list_to_adjacency_list,
    _normalize_numeric_argument,
    _normalize_color_argument,
    _normalize_shape_argument,
    _rescale_dict_values,
    _check_completeness,
    _get_optimal_offsets,
    _resample_spline,
    _get_total_pixels,
)
from ._node_layout import (
    get_fruchterman_reingold_layout,
    get_random_layout,
    get_sugiyama_layout,
    get_radial_tree_layout,
    get_circular_layout,
    get_linear_layout,
    get_bipartite_layout,
    get_multipartite_layout,
    get_shell_layout,
    get_community_layout,
    get_geometric_layout,
    _reduce_node_overlap,
    _remove_node_overlap,
)
from ._edge_layout import (
    _shift_edge,
    StraightEdgeLayout,
    CurvedEdgeLayout,
    BundledEdgeLayout,
    ArcDiagramEdgeLayout,
)

from ._artists import (
    Path,
    NodeArtist,
    CircularNodeArtist,
    RegularPolygonNodeArtist,
    EdgeArtist,
)
from ._parser import parse_graph, _parse_edge_list


BASE_SCALE = 1e-2
DEFAULT_COLOR = '#2c404c' # '#677e8c' # '#121f26' # '#23343f' # 'k',


class BaseGraph(object):
    """The Graph base class.

    Parameters
    ----------
    edges : list
        The edges of the graph, with each edge being represented by a (source node ID, target node ID) tuple.
    nodes : list or None, default None
        List of nodes. Required argument if any node in the graph is unconnected.
        If None, `nodes` is initialised to the set of the flattened `edges`.
    node_layout : str or dict, default 'spring'
        If `node_layout` is a string, the node positions are computed using the indicated method:

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

        If `node_layout` is a dict, keys are nodes and values are (x, y) positions.
    node_layout_kwargs : dict or None, default None
        Keyword arguments passed to node layout functions.
        See the documentation of the following functions for a full description of available options:

        - get_random_layout
        - get_circular_layout
        - get_fruchterman_reingold_layout
        - get_sugiyama_layout
        - get_radial_tree_layout
        - get_community_layout
        - get_bipartite_layout
        - get_multipartite_layout
        - get_shell_layout
        - get_geometric_layout

    node_shape : str or dict, default 'o'
        Node shape.
        If the type is str, all nodes have the same shape.
        If the type is dict, maps each node to an individual string representing the shape.
        The string specification is as for matplotlib.scatter marker, i.e. one of 'so^>v<dph8'.
    node_size : float or dict, default 3.
        Node size (radius).
        If the type is float, all nodes will have the same size.
        If the type is dict, maps each node to an individual size.

        .. note:: Values are rescaled by BASE_SCALE (1e-2) to be compatible with layout routines in igraph and networkx.

    node_edge_width : float or dict, default 0.5
        Line width of node marker border.
        If the type is float, all nodes have the same line width.
        If the type is dict, maps each node to an individual line width.

        .. note:: Values are rescaled by BASE_SCALE (1e-2) to be compatible with layout routines in igraph and networkx.

    node_color : matplotlib color specification or dict, default 'w'
        Node color.
        If the type is a string or RGBA array, all nodes have the same color.
        If the type is dict, maps each node to an individual color.
    node_edge_color : matplotlib color specification or dict, default {DEFAULT_COLOR}
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

        - size (adjusted to fit into node artists if offset is (0, 0))
        - horizontalalignment (default here: 'center')
        - verticalalignment (default here: 'center')
        - clip_on (default here: False)
        - zorder (default here: inf)

    edge_width : float or dict, default 1.
        Width of edges.
        If the type is a float, all edges have the same width.
        If the type is dict, maps each edge to an individual width.

        .. note:: Value is rescaled by BASE_SCALE (1e-2) to be compatible with layout routines in igraph and networkx.

    edge_color : matplotlib color specification or dict, default {DEFAULT_COLOR}
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
        - 'arc'      : draw edges as arcs with a fixed curvature
        - 'bundled'  : draw edges as edge bundles

        If edge_layout is a dict, the keys are edges and the values are edge paths
        in the form iterables of (x, y) tuples, the edge segments.
    edge_layout_kwargs : dict, default None
        Keyword arguments passed to edge layout functions.
        See the documentation of the following functions for a full description of available options:

        - get_straight_edge_paths
        - get_curved_edge_paths
        - get_bundled_edge_paths

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

        - horizontalalignment (default here: 'center'),
        - verticalalignment (default here: 'center')
        - clip_on (default here: False),
        - bbox (default here: dict(boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)),
        - zorder (default here: inf),
        - rotation (determined by edge_label_rotate argument)

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
    Graph, InteractiveGraph

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
        self.ax = self._initialize_axis(ax)
        self.fig = self.ax.get_figure()

        self.edges = self._initialize_edges(edges)
        self.nodes = self._initialize_nodes(nodes)

        self._raise_warning_if_graph_too_large()

        # Convert all node and edge parameters to dictionaries.
        node_shape      = _normalize_shape_argument(node_shape, self.nodes, 'node_shape')
        node_size       = _normalize_numeric_argument(node_size, self.nodes, 'node_size')
        node_edge_width = _normalize_numeric_argument(node_edge_width, self.nodes, 'node_edge_width')
        node_color      = _normalize_color_argument(node_color, self.nodes, 'node_color')
        node_edge_color = _normalize_color_argument(node_edge_color, self.nodes, 'node_edge_color')
        node_alpha      = _normalize_numeric_argument(node_alpha, self.nodes, 'node_alpha')
        node_zorder     = _normalize_numeric_argument(node_zorder, self.nodes, 'node_zorder')
        edge_width      = _normalize_numeric_argument(edge_width, self.edges, 'edge_width')
        edge_color      = _normalize_color_argument(edge_color, self.edges, 'edge_color')
        edge_alpha      = _normalize_numeric_argument(edge_alpha, self.edges, 'edge_alpha')
        edge_zorder     = _normalize_numeric_argument(edge_zorder, self.edges, 'edge_zorder')

        # Rescale.
        node_size       = _rescale_dict_values(node_size, BASE_SCALE)
        node_edge_width = _rescale_dict_values(node_edge_width, BASE_SCALE)
        edge_width      = _rescale_dict_values(edge_width, BASE_SCALE)

        self.node_size = node_size
        self.edge_width = edge_width
        self.origin = origin
        self.scale = scale

        # Initialise node and edge layouts and draw elements.
        self.node_positions = self._initialize_node_layout(
            node_layout, node_layout_kwargs)
        self.node_artists = dict()
        self.draw_nodes(self.nodes, self.node_positions,
                        node_shape, self.node_size, node_edge_width,
                        node_color, node_edge_color, node_alpha, node_zorder)

        self.edge_paths, self.edge_layout = self._initialize_edge_layout(
            edge_layout, edge_layout_kwargs)
        self.edge_artists = dict()
        self.draw_edges(self.edges, self.edge_paths, edge_width, edge_color, edge_alpha,
                        edge_zorder, arrows, self.node_artists)

        # This function needs to be called before any font sizes are adjusted,
        # as the axis dimensions affect the effective font size.
        self.ax.autoscale_view()
        if prettify:
            _make_pretty(self.ax)

        if node_labels:
            if isinstance(node_labels, bool):
                node_labels = dict(zip(self.nodes, self.nodes))
            self.node_label_fontdict = self._initialize_node_label_fontdict(node_label_fontdict)
            self.autoscale_node_labels = self._set_autoscale_node_labels_flag(self.node_label_fontdict, node_label_offset)
            self.node_label_offset, self._recompute_node_label_offsets =\
                self._initialize_node_label_offset(node_labels, node_label_offset)
            if self._recompute_node_label_offsets:
                self._update_node_label_offsets()
            self.node_label_artists = dict()
            self.draw_node_labels(node_labels, self.node_label_fontdict)
            if self.autoscale_node_labels:
                self._rescale_node_labels()
                self.fig.canvas.mpl_connect('resize_event', self._on_resize)

        if edge_labels:
            if isinstance(edge_labels, bool):
                edge_labels = dict(zip(self.edges, self.edges))
            self.edge_label_fontdict = self._initialize_edge_label_fontdict(edge_label_fontdict)
            self.edge_label_position = edge_label_position
            self.edge_label_rotate = edge_label_rotate
            self.edge_label_artists = dict()
            self.draw_edge_labels(edge_labels, self.edge_label_position,
                                  self.edge_label_rotate, self.edge_label_fontdict)


    def _initialize_edges(self, edges):
        return _parse_edge_list(edges)


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


    def _raise_warning_if_graph_too_large(self):
        total_pixels_per_edge = _get_total_pixels(self.fig) / len(self.edges)
        if total_pixels_per_edge < 400:
            msg = "The graph may be too large to visualize meaningfully."
            msg += f" There are only available {int(total_pixels_per_edge)} pixels per edge."
            msg += " This number is unlikely to be sufficient to render the edges without many overlaps, resulting in a 'hairball'."
            msg += " For comparison, each letter in this sentence is likely rendered using 200-400 pixels."
            import warnings
            warnings.warn(msg)


    def _initialize_node_layout(self, node_layout, node_layout_kwargs):
        if node_layout_kwargs is None:
            node_layout_kwargs = dict()

        if isinstance(node_layout, str):
            if (node_layout == 'spring') or (node_layout == 'dot') or (node_layout == 'radial'):
                node_layout_kwargs.setdefault('node_size', self.node_size)
            return self._get_node_positions(
                node_layout, node_layout_kwargs, self.edges,
            )

        elif isinstance(node_layout, dict):
            _check_completeness(set(node_layout), set(self.nodes), 'node_layout')
            # TODO check that nodes are within bounding box set by origin and scale
            return node_layout


    def _get_node_positions(self, node_layout, node_layout_kwargs, edges):
        if len(self.nodes) == 1:
            return {self.nodes[0]: np.array([self.origin[0] + 0.5 * self.scale[0], self.origin[1] + 0.5 * self.scale[1]])}
        if node_layout == 'spring':
            node_positions = get_fruchterman_reingold_layout(
                edges, nodes=self.nodes, origin=self.origin, scale=self.scale, **node_layout_kwargs)
            if len(node_positions) > 3: # Qhull fails for 2 or less nodes
                node_positions = _remove_node_overlap(node_positions, node_size=self.node_size, origin=self.origin, scale=self.scale)
            return node_positions
        elif node_layout == 'community':
            node_positions = get_community_layout(
                edges, nodes=self.nodes, origin=self.origin, scale=self.scale, **node_layout_kwargs)
            if len(node_positions) > 3: # Qhull fails for 2 or less nodes
                node_positions = _remove_node_overlap(node_positions, node_size=self.node_size, origin=self.origin, scale=self.scale)
            return node_positions
        elif node_layout == 'circular':
            return get_circular_layout(
                edges, nodes=self.nodes, origin=self.origin, scale=self.scale, **node_layout_kwargs)
        elif node_layout == 'linear':
            return get_linear_layout(
                edges, nodes=self.nodes, origin=self.origin, scale=self.scale, **node_layout_kwargs)
        elif node_layout == 'bipartite':
            return get_bipartite_layout(
                edges, nodes=self.nodes, origin=self.origin, scale=self.scale, **node_layout_kwargs)
        elif node_layout == 'multipartite':
            return get_multipartite_layout(
                edges, origin=self.origin, scale=self.scale, **node_layout_kwargs)
        elif node_layout == 'shell':
            return get_shell_layout(
                edges, origin=self.origin, scale=self.scale, **node_layout_kwargs)
        elif node_layout == 'dot':
            return get_sugiyama_layout(
                edges, nodes=self.nodes, origin=self.origin, scale=self.scale, **node_layout_kwargs)
        elif node_layout == 'radial':
            return get_radial_tree_layout(
                edges, nodes=self.nodes, origin=self.origin, scale=self.scale, **node_layout_kwargs)
        elif node_layout == 'random':
            return get_random_layout(
                edges, nodes=self.nodes, origin=self.origin, scale=self.scale, **node_layout_kwargs)
        elif node_layout == 'geometric':
            return get_geometric_layout(
                edges, nodes=self.nodes, origin=self.origin, scale=self.scale, **node_layout_kwargs)
        else:
            implemented = ['spring', 'community', 'circular',
                           'linear', 'bipartite', 'multipartite',
                           'shell', 'dot', 'radial', 'random',
                           'geometric']
            msg = f"Node layout {node_layout} not implemented. Available layouts are:"
            for method in implemented:
                msg += f"\n\t{method}"
            raise NotImplementedError(msg)


    def _initialize_edge_layout(self, edge_layout, edge_layout_kwargs):

        if edge_layout_kwargs is None:
            edge_layout_kwargs = dict()

        selfloops = [(source, target) for (source, target) in self.edges if source==target]
        if selfloops:
            if 'selfloop_radius' in edge_layout_kwargs:
                edge_layout_kwargs['selfloop_radius'] = \
                    _normalize_numeric_argument(edge_layout_kwargs['selfloop_radius'], selfloops, 'selfloop_radius')
            else:
                edge_layout_kwargs['selfloop_radius'] = \
                    {(node, node) : 1.5 * self.node_size[node] for node, _ in selfloops}

        if isinstance(edge_layout, str):
            if edge_layout == "straight":
                edge_layout = StraightEdgeLayout(self.edges, self.node_positions, **edge_layout_kwargs)
            elif edge_layout == "curved":
                edge_layout_kwargs.setdefault('node_size', self.node_size)
                edge_layout_kwargs.setdefault('origin', self.origin)
                edge_layout_kwargs.setdefault('scale', self.scale)
                edge_layout = CurvedEdgeLayout(self.edges, self.node_positions, **edge_layout_kwargs)
            elif edge_layout == "bundled":
                edge_layout = BundledEdgeLayout(self.edges, self.node_positions, **edge_layout_kwargs)
            elif edge_layout == "arc":
                edge_layout = ArcDiagramEdgeLayout(self.edges, self.node_positions, **edge_layout_kwargs)
            else:
                raise NotImplementedError(f"Variable edge_layout one of 'straight', 'curved', 'bundled', or 'arc', not {edge_layout}")
            edge_paths = edge_layout.compute()

        elif isinstance(edge_layout, dict):
            _check_completeness(edge_layout, self.edges, 'edge_layout')
            # TODO check that edge paths are within bounding box given by origin and scale
            edge_paths = edge_layout
            edge_layout = StraightEdgeLayout(self.edges, self.node_positions, **edge_layout_kwargs)
            edge_layout.edge_paths.update(edge_paths)

        else:
            raise TypeError("Variable `edge_layout` either a string or a dict mapping edges to edge paths.")

        return edge_paths, edge_layout


    def _initialize_axis(self, ax):
        if ax is None:
            return plt.gca()
        elif isinstance(ax, mpl.axes.Axes):
            return ax
        else:
            raise TypeError(f"Variable 'ax' either None or a matplotlib axis instance. However, type(ax) is {type(ax)}.")


    def draw_nodes(self, nodes, node_positions, node_shape, node_size,
                   node_edge_width, node_color, node_edge_color, node_alpha,
                   node_zorder):
        """Draw or update node artists.

        Parameters
        ----------
        nodes : list
            List of nodes IDs.
        node_positions : dict
            Mapping of nodes to (x, y) positions.
        node_shape : dict
            Mapping of nodes to shapes.
            Specification is as for matplotlib.scatter marker, i.e. one of 'so^>v<dph8'.
        node_size : dict
            Mapping of nodes to sizes.
        node_edge_width : dict
            Mapping of nodes to marker edge widths.
        node_color : dict
            Mapping of nodes to valid matplotlib color specifications.
        node_edge_color : dict
            Mapping of nodes to valid matplotlib color specifications.
        node_alpha : dict
            Mapping of nodes to node transparencies.
        node_zorder : dict
            Mapping of nodes to z-orders.

        Returns
        -------
        node_artists: dict
            Updates mapping of nodes to corresponding node artists.

        """

        for node in nodes:

            kwargs = dict(
                xy        = node_positions[node],
                size      = node_size[node],
                facecolor = node_color[node],
                edgecolor = node_edge_color[node],
                linewidth = node_edge_width[node],
                alpha     = node_alpha[node],
                zorder    = node_zorder[node],
            )

            shape = node_shape[node]

            if isinstance(shape, Path):
                node_artist = NodeArtist(shape, **kwargs)

            elif isinstance(shape, str):
                if shape == 'o':
                    node_artist = CircularNodeArtist(**kwargs)
                elif shape in 's^>v<dph8':
                    symbol_to_parameters = {
                        '^' : (3, 0),
                        '<' : (3, np.pi * 0.5),
                        'v' : (3, np.pi),
                        '>' : (3, np.pi * 1.5),
                        's' : (4, np.pi * 0.25),
                        'd' : (4, np.pi * 0.5),
                        'p' : (5, 0),
                        'h' : (6, 0),
                        '8' : (8, 0),
                    }
                    node_artist = RegularPolygonNodeArtist(*symbol_to_parameters[shape], **kwargs)
                else:
                    raise ValueError("Node shape should be one of: 'so^>v<dph8'. Current shape:{}".format(shape))
            else:
                raise ValueError("Node shape should be a matplotlob.Path instance or one of: 'so^>v<dph8'. Current shape:{}".format(shape))

            self.ax.add_patch(node_artist)

            if node in self.node_artists:
                self.node_artists[node].remove()
            self.node_artists[node] = node_artist


    def _update_node_artists(self, nodes):
        for node in nodes:
            self.node_artists[node].xy = self.node_positions[node]


    def draw_edges(self, edges, edge_path, edge_width, edge_color, edge_alpha,
                   edge_zorder, arrows, node_artists):
        """Draw or update edge artists.

        Parameters
        ----------
        edges : list
            The edges of the graph, with each edge being represented by a (source node ID, target node ID) tuple.
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

        for edge in edge_path:

            curved = False if (len(edge_path[edge]) == 2) else True

            source, target = edge
            if ((target, source) in edge_path) and (source != target): # i.e. bidirectional edges excluding self-loops
                if np.allclose(edge_path[(source, target)], edge_path[(target, source)][::-1]): # i.e. same path
                    shape = 'right' # i.e. plot half arrow / thin line shifted to the right
                else:
                    shape = 'full'
            else:
                shape = 'full'

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


    def _initialize_node_label_fontdict(self, node_label_fontdict):
        if node_label_fontdict is None:
            node_label_fontdict = dict()

        node_label_fontdict.setdefault('clip_on', False)
        node_label_fontdict.setdefault('zorder', np.inf)

        if 'ha' not in node_label_fontdict:
            node_label_fontdict.setdefault('horizontalalignment', 'center')

        if 'va' not in node_label_fontdict:
            node_label_fontdict.setdefault('verticalalignment', 'center')

        return node_label_fontdict


    def _set_autoscale_node_labels_flag(self, node_label_fontdict, node_label_offset):
        flag = False
        if ('size' not in node_label_fontdict) and ('fontsize' not in node_label_fontdict):
            if (node_label_fontdict.get('verticalalignment', 'center') == 'center') and \
               (node_label_fontdict.get('va', 'center') == 'center') and \
               (node_label_fontdict.get('horizonaltalignment', 'center') == 'center') and \
               (node_label_fontdict.get('ha', 'center') == 'center'):
                if np.all(np.isclose(node_label_offset, (0, 0))): # Labels are centered on node artists.
                    flag = True
        return flag


    def draw_node_labels(self, node_labels, node_label_fontdict):
        """Draw or update node labels.

        Parameters
        ----------
        node_labels : dict
           Mapping of nodes to strings, the node labels.
           Only nodes in the dictionary are labelled.
        node_label_offset: tuple, default (0., 0.)
            The (x, y) offset from node centre of label position.
        node_label_fontdict : dict
            Keyword arguments passed to matplotlib.text.Text.
            For a full list of available arguments see the matplotlib documentation.
            The following default values differ from the defaults for matplotlib.text.Text:
                - size (adjusted to fit into node artists if offset is (0, 0))
                - horizontalalignment (default here: 'center')
                - verticalalignment (default here: 'center')
                - clip_on (default here: False)

        Returns
        -------
        self.node_label_artists: dict
            Updates mapping of nodes to text objects, the node label artists.

        """

        for node, label in node_labels.items():
            x, y = self.node_positions[node]
            dx, dy = self.node_label_offset[node]
            artist = self.ax.text(x+dx, y+dy, label, **node_label_fontdict)

            if node in self.node_label_artists:
                self.node_label_artists[node].remove()
            self.node_label_artists[node] = artist


    def _rescale_node_labels(self, fudge_factor=0.75):
        maximum_font_sizes = [self.node_artists[node].get_maximum_fontsize(text_object) \
                              for node, text_object in self.node_label_artists.items()]
        font_size = np.min(maximum_font_sizes)
        for text_object in self.node_label_artists.values():
            text_object.set_size(fudge_factor * font_size)


    def _maximize_node_labels(self, fudge_factor=0.75):
        for node, text_object in self.node_label_artists.items():
            font_size = self.node_artists[node].get_maximum_fontsize(text_object)
            text_object.set_size(fudge_factor * font_size)


    def _on_resize(self, event):
        self._rescale_node_labels()


    def _update_node_label_positions(self):
        if self._recompute_node_label_offsets:
            self._update_node_label_offsets()

        for node, (dx, dy) in self.node_label_offset.items():
            x, y = self.node_positions[node]
            self.node_label_artists[node].set_position((x + dx, y + dy))


    def _update_node_label_offsets(self, total_samples_per_edge=100):
        anchors = np.array([self.node_positions[node] for node in self.node_label_offset.keys()])
        offsets = np.array(list(self.node_label_offset.values()))
        avoid = np.concatenate([_resample_spline(path, total_samples_per_edge) for path in self.edge_paths.values()], axis=0)
        optimal_offsets = _get_optimal_offsets(anchors, offsets, avoid)

        for ii, node in enumerate(self.node_label_offset):
            self.node_label_offset[node] = optimal_offsets[ii]


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
        """Draw or update edge labels.

        Parameters
        ----------
        edge_labels : dict
            Mapping of edges to strings, the edge labels.
            Only edges in the dictionary are labelled.
        edge_label_position : float
            Relative position along the edge where the label is placed.
                head   : 0.
                centre : 0.5
                tail   : 1.
        edge_label_rotate : bool
            If True, edge labels are rotated such that they have the same orientation as their corresponding edge.
            If False, edge labels are not rotated; the angle of the text is parallel to the axis.
        edge_label_fontdict : dict
            Keyword arguments passed to matplotlib.text.Text.

        Returns
        -------
        self.edge_label_artists: dict
            Updates mapping of edges to text objects, the edge label artists.

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


class Graph(BaseGraph):
    """Parses the given graph data object and initialises the BaseGraph object.

    If the given graph includes edge weights, then these are mapped to colors using the `edge_cmap` parameter.

    Parameters
    ----------
    graph : various formats
        Graph object to plot. Various input formats are supported.
        In order of precedence:

        - Edge list:
          Iterable of (source, target) or (source, target, weight) tuples,
          or equivalent (E, 2) or (E, 3) ndarray, where E is the number of edges.
        - Adjacency matrix:
          Full-rank (V, V) ndarray, where V is the number of nodes/vertices.
          The absence of a connection is indicated by a zero.

          .. note:: If V <= 3, any (2, 2) or (3, 3) matrices will be interpreted as edge lists.**

        - networkx.Graph, igraph.Graph, or graph_tool.Graph object

    node_layout : str or dict, default 'spring'
        If `node_layout` is a string, the node positions are computed using the indicated method:

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

        If `node_layout` is a dict, keys are nodes and values are (x, y) positions.
    node_layout_kwargs : dict or None, default None
        Keyword arguments passed to node layout functions.
        See the documentation of the following functions for a full description of available options:

        - get_random_layout
        - get_circular_layout
        - get_fruchterman_reingold_layout
        - get_sugiyama_layout
        - get_radial_tree_layout
        - get_community_layout
        - get_bipartite_layout
        - get_multipartite_layout
        - get_shell_layout
        - get_geometric_layout

    node_shape : str or dict, default 'o'
        Node shape.
        If the type is str, all nodes have the same shape.
        If the type is dict, maps each node to an individual string representing the shape.
        The string specification is as for matplotlib.scatter marker, i.e. one of 'so^>v<dph8'.
    node_size : float or dict, default 3.
        Node size (radius).
        If the type is float, all nodes will have the same size.
        If the type is dict, maps each node to an individual size.

        .. note:: Values are rescaled by BASE_SCALE (1e-2) to be compatible with layout routines in igraph and networkx.

    node_edge_width : float or dict, default 0.5
        Line width of node marker border.
        If the type is float, all nodes have the same line width.
        If the type is dict, maps each node to an individual line width.

        .. note: Values are rescaled by BASE_SCALE (1e-2) to be compatible with layout routines in igraph and networkx.

    node_color : matplotlib color specification or dict, default 'w'
        Node color.
        If the type is a string or RGBA array, all nodes have the same color.
        If the type is dict, maps each node to an individual color.
    node_edge_color : matplotlib color specification or dict, default DEFAULT_COLOR
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

        - size (adjusted to fit into node artists if offset is (0, 0))
        - horizontalalignment (default here: 'center')
        - verticalalignment (default here: 'center')
        - clip_on (default here: False)
        - zorder (default here: inf)

    edge_width : float or dict, default 1.
        Width of edges.
        If the type is a float, all edges have the same width.
        If the type is dict, maps each edge to an individual width.

        .. note:: Value is rescaled by BASE_SCALE (1e-2) to be compatible with layout routines in igraph and networkx.

    edge_cmap : matplotlib color map (default 'RdGy')
        Color map used to map edge weights to edge colors. Should be diverging.
        If edge weights are strictly positive, weights are mapped to the
        left hand side of the color map with vmin=0 and vmax=np.max(weights).
        If edge weights are positive and negative, then weights are mapped
        to colors such that a weight of zero corresponds to the center of the
        color map; the boundaries are set to +/- the maximum absolute weight.
        If the graph is unweighted or the edge colors are specified explicitly,
        this parameter is ignored.
    edge_color : matplotlib color specification or dict, default DEFAULT_COLOR
        Edge color. If provided explicitly, overrides `edge_cmap`.
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

        If edge_layout is a dict, the keys are edges and the values are edge paths
        in the form iterables of (x, y) tuples, the edge segments.
    edge_layout_kwargs : dict, default None
        Keyword arguments passed to edge layout functions.
        See the documentation of the following functions for a full description of available options:
        - get_straight_edge_paths
        - get_curved_edge_paths
        - get_bundled_edge_paths
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

        - horizontalalignment (default here: 'center'),
        - verticalalignment (default here: 'center')
        - clip_on (default here: False),
        - bbox (default here: dict(boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)),
        - zorder (default here: inf),
        - rotation (determined by edge_label_rotate argument)

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
    BaseGraph, InteractiveGraph

    """

    def __init__(self, graph, edge_cmap='RdGy', *args, **kwargs):

        # Accept a variety of formats for 'graph' and convert to common denominator.
        nodes, edges, edge_weight = self._parse_input(graph)
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


    def _parse_input(self, graph):
        return parse_graph(graph)


def _get_color(mydict, cmap='RdGy', vmin=None, vmax=None):
    """Map positive and negative floats to a diverging colormap, such that
    - the midpoint of the colormap corresponds to a value of 0., and
    - values above and below the midpoint are mapped linearly and in equal measure
      to increases in color intensity.

    Parameters
    ----------
    mydict: dict
        Mapping of graph element (node, edge) to a float.
        For example (source, target) : edge weight.
    cmap: str, default 'RdGy'
        Matplotlib colormap specification.
    vmin, vmax: float or None, default None
        Minimum and maximum float corresponding to the dynamic range of the colormap.

    Returns
    -------
    newdict: dict
        Mapping of graph elements to RGBA tuples.

    """

    keys = mydict.keys()
    values = np.array(list(mydict.values()), dtype=float)

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
    mapper = mpl.cm.ScalarMappable(cmap=cmap)
    mapper.set_clim(vmin=0., vmax=1.)
    colors = mapper.to_rgba(values)

    return {key: color for (key, color) in zip(keys, colors)}


def _get_zorder(color_dict):
    """Reorder plot elements such that darker items are plotted last and hence most prominent in the graph.
    This assumes that the background is white.

    """
    intensities = [rgba_to_grayscale(*v) for v in color_dict.values()]
    zorder = _rank(intensities)
    zorder = np.max(zorder) - zorder # reverse order as greater values correspond to lighter colors
    return {key: index for key, index in zip(color_dict.keys(), zorder)}


def rgba_to_grayscale(r, g, b, a=1):
    """Convert RGBA values to grayscale.

    Notes
    -----
    Adapted from: https://stackoverflow.com/a/689547/2912349

    """

    return (0.299 * r + 0.587 * g + 0.114 * b) * a


class ClickableArtists(object):
    """Implements selection of matplotlib artists via the mouse left click (+/- ctrl or command key).

    Notes:
    ------
    Adapted from: https://stackoverflow.com/a/47312637/2912349

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

        if mpl.get_backend() == 'MacOSX':
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
    Adapted from: https://stackoverflow.com/a/47312637/2912349

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
        self._rect.set_xy((xlim[0], ylim[0]))
        self._rect.set_width(xlim[1] - xlim[0])
        self._rect.set_height(ylim[1] - ylim[0])
        self.fig.canvas.draw_idle()


class DraggableArtists(SelectableArtists):
    """Augments SelectableArtists to support dragging of artists by holding the left mouse button.

    Notes:
    ------
    Adapted from: https://stackoverflow.com/a/47312637/2912349

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
    """Augments `Graph` to support selection and dragging of node artists with the mouse."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_dragging_clicking_and_selecting()


    def _setup_dragging_clicking_and_selecting(self):
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

        edge_paths = self.edge_layout.update(nodes)
        self._update_edge_artists(edge_paths)
        if hasattr(self, 'edge_label_artists'):
            self._update_edge_label_positions(edge_paths)

        self.fig.canvas.draw_idle()


    def _get_stale_nodes(self):
        return [self._draggable_artist_to_node[artist] for artist in self._selected_artists if artist in self._draggable_artists]


    def _update_node_positions(self, nodes, cursor_position):
        for node in nodes:
            self.node_positions[node] = cursor_position + self._offset[self.node_artists[node]]


    def _on_release(self, event):
        if self._currently_dragging:
            edge_paths = self.edge_layout.compute()
            self._update_edge_artists(edge_paths)
            if hasattr(self, 'edge_label_artists'): # move edge labels
                self._update_edge_label_positions(edge_paths)

        super()._on_release(event)


    # def _on_resize(self, event):
    #     if hasattr(self, 'node_labels'):
    #         self.draw_node_labels(self.node_labels)
    #         # print("As node label font size was not explicitly set, automatically adjusted node label font size to {:.2f}.".format(self.node_label_font_size))


class DraggableGraphWithGridMode(DraggableGraph):
    """
    Implements a grid-mode, in which node positions are fixed to a grid.
    To activate, press the letter 'g'.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_grid_mode()


    def _setup_grid_mode(self):
        self.grid = False
        self.grid_dx = 0.05 * self.scale[0]
        self.grid_dy = 0.05 * self.scale[1]
        self._grid_lines = []
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_toggle)


    def _on_key_toggle(self, event):
        if event.key == 'g':
            if self.grid is False:
                self.grid = True
                self._draw_grid()
            else:
                self.grid = False
                self._remove_grid()
        self.fig.canvas.draw_idle()


    def _draw_grid(self):
        eps = 1e-13
        for x in np.arange(self.origin[0], self.origin[0] + self.scale[0] + eps, self.grid_dx):
            line = self.ax.axvline(x, color='k', alpha=0.1, linestyle='--')
            self._grid_lines.append(line)

        for y in np.arange(self.origin[1], self.origin[1] + self.scale[1] + eps, self.grid_dy):
            line = self.ax.axhline(y, color='k', alpha=0.1, linestyle='--')
            self._grid_lines.append(line)


    def _remove_grid(self):
        for line in self._grid_lines:
            line.remove()
        self._grid_lines = []


    def _on_release(self, event):
        if self._currently_dragging and self.grid:
            nodes = self._get_stale_nodes()
            for node in nodes:
                self.node_positions[node] = self._get_nearest_grid_coordinate(*self.node_positions[node])
            self._update_node_artists(nodes)
            if hasattr(self, 'node_label_artists'):
                self._update_node_label_positions()

            edge_paths = self.edge_layout.compute()
            self._update_edge_artists(edge_paths)
            if hasattr(self, 'edge_label_artists'): # move edge labels
                self._update_edge_label_positions(edges)

        super()._on_release(event)


    def _get_nearest_grid_coordinate(self, x, y):
        x = np.round((x - self.origin[0]) / self.grid_dx) * self.grid_dx + self.origin[0]
        y = np.round((y - self.origin[1]) / self.grid_dy) * self.grid_dy + self.origin[1]
        return x, y


class EmphasizeOnHover(object):
    """Emphasize matplotlib artists when hovering over them by desaturating all other artists."""

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
    """Combines `EmphasizeOnHover` with the `Graph` class such that nodes are emphasized when hovering over them with the mouse.

    Parameters
    ----------
    graph : various formats
        Graph object to plot. Various input formats are supported.
        In order of precedence:

        - Edge list:
          Iterable of (source, target) or (source, target, weight) tuples,
          or equivalent (E, 2) or (E, 3) ndarray, where E is the number of edges.
        - Adjacency matrix:
          Full-rank (V, V) ndarray, where V is the number of nodes/vertices.
          The absence of a connection is indicated by a zero.

          .. note:: If V <= 3, any (2, 2) or (3, 3) matrices will be interpreted as edge lists.**

        - networkx.Graph, igraph.Graph, or graph_tool.Graph object

    mouseover_highlight_mapping : dict or None, default None
        Determines which nodes and/or edges are highlighted when hovering over any given node or edge.
        The keys of the dictionary are node and/or edge IDs, while the values are iterables of node and/or edge IDs.
        If the parameter is None, a default dictionary is constructed, which maps

        - edges to themselves as well as their source and target nodes, and
        - nodes to themselves as well as their immediate neighbours and any edges between them.

    *args, **kwargs
        Parameters passed through to `Graph`. See its documentation for a full list of available arguments.

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
    Graph

    """

    def __init__(self, graph, mouseover_highlight_mapping=None, *args, **kwargs):
        Graph.__init__(self, graph, *args, **kwargs)

        self._setup_emphasis()
        if mouseover_highlight_mapping is not None: # this includes empty mappings!
            self._check_mouseover_highlight_mapping(mouseover_highlight_mapping)
            self.mouseover_highlight_mapping = mouseover_highlight_mapping


    def _setup_emphasis(self):
        self.artist_to_key = self._map_artist_to_key()
        EmphasizeOnHover.__init__(self, list(self.artist_to_key.keys()))
        self.mouseover_highlight_mapping = self._get_default_mouseover_highlight_mapping()


    def _map_artist_to_key(self):
        artists = list(self.node_artists.values()) + list(self.edge_artists.values())
        keys = list(self.node_artists.keys()) + list(self.edge_artists.keys())
        return dict(zip(artists, keys))


    def _get_default_mouseover_highlight_mapping(self):
        """Construct default mapping:

        - Nodes map to themselves, their neighbours, and any edges between them.
        - Edges map to themselves, and their source and target node.

        """

        mapping = dict()

        for node in self.nodes:
            mapping[node] = [node]

        for edge in self.edges:
            source, target = edge[:2]
            mapping[source].append(edge)
            mapping[source].append(target)
            mapping[target].append(edge)
            mapping[target].append(source)
            mapping[edge] = [edge, source, target]

        return mapping


    def _check_mouseover_highlight_mapping(self, mapping):

        if not isinstance(mapping, dict):
            raise TypeError(f"Parameter `mouseover_highlight_mapping` is a dictionary, not {type(mapping)}.")

        invalid_keys = []
        for key in mapping:
            if key in self.node_artists:
                pass
            elif key in self.edge_artists:
                pass
            else:
                invalid_keys.append(key)
        if invalid_keys:
            msg = "Parameter `mouseover_highlight_mapping` contains invalid keys:"
            for key in invalid_keys:
                msg += f"\n\t- {key}"
            raise ValueError(msg)

        invalid_values = []
        for values in mapping.values():
            for value in values:
                if value in self.node_artists:
                    pass
                elif value in self.edge_artists:
                    pass
                else:
                    invalid_values.append(value)
        if invalid_values:
            msg = "Parameter `mouseover_highlight_mapping` contains invalid values:"
            for value in set(invalid_values):
                msg += f"\n\t- {value}"
            raise ValueError(msg)


    def _on_motion(self, event):

        if event.inaxes == self.ax:

            # determine if the cursor is on an artist
            selected_artist = None
            for artist in self.emphasizeable_artists:
                if artist.contains(event)[0]: # returns bool, {} for some reason
                    selected_artist = artist
                    break

            if selected_artist:
                key = self.artist_to_key[artist]
                if key in self.mouseover_highlight_mapping:
                    emphasized_artists = []
                    for value in self.mouseover_highlight_mapping[key]:
                        if value in self.node_artists:
                            emphasized_artists.append(self.node_artists[value])
                        elif value in self.edge_artists:
                            emphasized_artists.append(self.edge_artists[value])

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
    """Show or hide annotations when clicking on matplotlib artists."""

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

        if not hasattr(self, "fig"):
            try:
                self.fig, = set(list(artist.figure for artist in artist_to_annotation))
            except ValueError:
                raise Exception("All artists have to be on the same figure!")

        if not hasattr(self, "ax"):
            try:
                self.ax, = set(list(artist.axes for artist in artist_to_annotation))
            except ValueError:
                raise Exception("All artists have to be on the same axis!")

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
        x, y = artist.xy + 2 * artist.size * vector
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


    def _redraw_annotations(self, event):
        if event.inaxes == self.ax:
            for artist in self.annotated_artists:
                self._remove_annotation(artist)
                placement = self._get_annotation_placement(artist)
                self._add_annotation(artist, *placement)
            self.fig.canvas.draw()


class AnnotateOnClickGraph(Graph, AnnotateOnClick):
    """Combines `AnnotateOnClick` with the `Graph` class such that nodes or edges can have toggleable annotations."""

    def __init__(self, *args, **kwargs):
        Graph.__init__(self, *args, **kwargs)
        self._setup_annotations(*args, **kwargs)


    def _setup_annotations(self, *args, **kwargs):
        if "annotations" in kwargs:
            artist_to_annotation = self._map_artist_to_annotation(kwargs["annotations"])
        else:
            artist_to_annotation = dict()

        if "annotation_fontdict" in kwargs:
            AnnotateOnClick.__init__(self, artist_to_annotation, kwargs["annotation_fontdict"])
        else:
            AnnotateOnClick.__init__(self, artist_to_annotation)


    def _map_artist_to_annotation(self, annotations):
        artist_to_annotation = dict()
        for key, annotation in annotations.items():
            if key in self.nodes:
                artist_to_annotation[self.node_artists[key]] = annotation
            elif key in self.edges:
                artist_to_annotation[self.edge_artists[key]] = annotation
            else:
                raise ValueError(f"There is no node or edge with the ID {key} for the annotation '{annotation}'.")
        return artist_to_annotation


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
    """Show or hide tabular information when clicking on matplotlib artists."""

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

        if not hasattr(self, "fig"):
            try:
                self.fig, = set(list(artist.figure for artist in artist_to_table))
            except ValueError:
                raise Exception("All artists have to be on the same figure!")

        if not hasattr(self, "ax"):
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
    """Combines `TableOnClick` with the `Graph` class such that nodes or edges can have toggleable tabular annotations."""

    def __init__(self, *args, **kwargs):
        Graph.__init__(self, *args, **kwargs)
        self._setup_table_annotations(*args, **kwargs)


    def _setup_table_annotations(self, *args, **kwargs):
        if 'tables' in kwargs:
            artist_to_table = _map_artist_to_table(kwargs['tables'])
        else:
            artist_to_table = dict()

        if 'table_kwargs' in kwargs:
            TableOnClick.__init__(self, artist_to_table, kwargs['table_kwargs'])
        else:
            TableOnClick.__init__(self, artist_to_table)


    def _map_artist_to_table(self, tables):
        artist_to_table = dict()
        for key, table in tables.items():
            if key in self.nodes:
                artist_to_table[self.node_artists[key]] = table
            elif key in self.edges:
                artist_to_table[self.edge_artists[key]] = table
            else:
                raise ValueError(f"There is no node or edge with the ID {key} for the table '{table}'.")
        return artist_to_table


class InteractiveGraph(DraggableGraphWithGridMode, EmphasizeOnHoverGraph, AnnotateOnClickGraph, TableOnClickGraph):
    """Extends the `Graph` class to support node placement with the mouse, emphasis of graph elements when hovering over them, and toggleable annotations.

    - Nodes can be selected and dragged around with the mouse.
    - Nodes and edges are emphasized when hovering over them.
    - Supports additional annotations that can be toggled on and off by clicking on the corresponding node or edge.
    - These annotations can also be tables.

    Parameters
    ----------
    graph : various formats
        Graph object to plot. Various input formats are supported.
        In order of precedence:

        - Edge list:
          Iterable of (source, target) or (source, target, weight) tuples,
          or equivalent (E, 2) or (E, 3) ndarray, where E is the number of edges.
        - Adjacency matrix:
          Full-rank (V, V) ndarray, where V is the number of nodes/vertices.
          The absence of a connection is indicated by a zero.

          .. note:: If V <= 3, any (2, 2) or (3, 3) matrices will be interpreted as edge lists.

        - networkx.Graph, igraph.Graph, or graph_tool.Graph object

    node_layout : str or dict, default 'spring'
        If `node_layout` is a string, the node positions are computed using the indicated method:

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

        If `node_layout` is a dict, keys are nodes and values are (x, y) positions.
    node_layout_kwargs : dict or None, default None
        Keyword arguments passed to node layout functions.
        See the documentation of the following functions for a full description of available options:

        - get_random_layout
        - get_circular_layout
        - get_fruchterman_reingold_layout
        - get_sugiyama_layout
        - get_radial_tree_layout
        - get_community_layout
        - get_bipartite_layout
        - get_multipartite_layout
        - get_shell_layout
        - get_geometric_layout

    node_shape : str or dict, default 'o'
        Node shape.
        If the type is str, all nodes have the same shape.
        If the type is dict, maps each node to an individual string representing the shape.
        The string specification is as for matplotlib.scatter marker, i.e. one of 'so^>v<dph8'.
    node_size : float or dict, default 3.
        Node size (radius).
        If the type is float, all nodes will have the same size.
        If the type is dict, maps each node to an individual size.

        .. note:: Values are rescaled by BASE_SCALE (1e-2) to be compatible with layout routines in igraph and networkx.

    node_edge_width : float or dict, default 0.5ayout        Line width of node marker border.
        If the type is float, all nodes have the same line width.
        If the type is dict, maps each node to an individual line width.

        ..note:: Values are rescaled by BASE_SCALE (1e-2) to be compatible with layout routines in igraph and networkx.

    node_color : matplotlib color specification or dict, default 'w'
        Node color.
        If the type is a string or RGBA array, all nodes have the same color.
        If the type is dict, maps each node to an individual color.
    node_edge_color : matplotlib color specification or dict, default DEFAULT_COLOR
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

        - size (adjusted to fit into node artists if offset is (0, 0))
        - horizontalalignment (default here: 'center')
        - verticalalignment (default here: 'center')
        - clip_on (default here: False)
        - zorder (default here: inf)

    edge_width : float or dict, default 1.
        Width of edges.
        If the type is a float, all edges have the same width.
        If the type is dict, maps each edge to an individual width.

        .. note:: Value is rescaled by BASE_SCALE (1e-2) to be compatible with layout routines in igraph and networkx.

    edge_cmap : matplotlib color map (default 'RdGy')
        Color map used to map edge weights to edge colors. Should be diverging.
        If edge weights are strictly positive, weights are mapped to the
        left hand side of the color map with vmin=0 and vmax=np.max(weights).
        If edge weights are positive and negative, then weights are mapped
        to colors such that a weight of zero corresponds to the center of the
        color map; the boundaries are set to +/- the maximum absolute weight.
        If the graph is unweighted or the edge colors are specified explicitly,
        this parameter is ignored.
    edge_color : matplotlib color specification or dict, default DEFAULT_COLOR
        Edge color. If provided explicitly, overrides `edge_cmap`.
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

        If edge_layout is a dict, the keys are edges and the values are edge paths
        in the form iterables of (x, y) tuples, the edge segments.
    edge_layout_kwargs : dict, default None
        Keyword arguments passed to edge layout functions.
        See the documentation of the following functions for a full description of available options:

        - get_straight_edge_paths
        - get_curved_edge_paths
        - get_bundled_edge_paths

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

        - horizontalalignment (default here: 'center'),
        - verticalalignment (default here: 'center')
        - clip_on (default here: False),
        - bbox (default here: dict(boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)),
        - zorder (default here: inf),
        - rotation (determined by edge_label_rotate argument)

    annotations : dict
        Mapping of nodes or edges to strings or dictionaries, the annotations.
        The visibility of the annotations can be toggled on or off by clicking on the corresponding node or edge.

        .. line-block::
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
        Mapping of nodes and/or edges to pandas dataframes.
        The visibility of the tables that can toggled on or off by clicking on the corresponding node or edge.
    table_kwargs : dict
        Keyword arguments passed to matplotlib.pyplot.table.
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
    Graph

    Notes
    -----
    You must retain a reference to the plot instance!
    Otherwise, the plot instance will be garbage collected after the initial draw
    and you won't be able to move the plot elements around.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from netgraph import InteractiveGraph
    >>> plt.ion()
    >>> plot_instance = InteractiveGraph(my_graph_obj)
    >>> plt.show()

    """

    def __init__(self, *args, **kwargs):
        DraggableGraphWithGridMode.__init__(self, *args, **kwargs)
        self._setup_emphasis()
        self._setup_annotations(*args, **kwargs)
        self._setup_table_annotations(*args, **kwargs)


    def _on_motion(self, event):
        DraggableGraphWithGridMode._on_motion(self, event)
        EmphasizeOnHoverGraph._on_motion(self, event)


    def _on_release(self, event):
        if self._currently_dragging is False:
            DraggableGraphWithGridMode._on_release(self, event)
            if self.artist_to_annotation:
                AnnotateOnClickGraph._on_release(self, event)
            if self.artist_to_table:
                TableOnClickGraph._on_release(self, event)
        else:
            DraggableGraphWithGridMode._on_release(self, event)
            if self.artist_to_annotation:
                self._redraw_annotations(event)
