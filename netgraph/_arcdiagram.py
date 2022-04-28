#!/usr/bin/env python
"""
Implements the ArcDiagram class and interactive and editable variants.
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import ConnectionStyle

from ._main import (
    BaseGraph,
    Graph,
    DraggableArtists,
    DraggableGraph,
    InteractiveGraph,
    EmphasizeOnHover,
    EmphasizeOnHoverGraph,
    AnnotateOnClick,
    AnnotateOnClickGraph,
    TableOnClick,
    TableOnClickGraph,
)
from ._interactive_variants import (
    MutableGraph,
    EditableGraph,
)
from ._node_layout import (
    get_linear_layout,
)
from ._edge_layout import (
    get_arced_edge_paths,
    get_selfloop_paths,
)
from ._utils import (
    _get_gradient_and_intercept,
    _is_above_line,
    _reflect_across_line,
    _bspline,
    _are_collinear,
    _get_orthogonal_projection_onto_segment,
)


def _lateralize_arced_edge_paths(edge_paths, node_positions, above):
    """Ensure that edge paths are all either above or below the line passing through nodes."""
    for (source, target), path in edge_paths.items():
        edge_paths[(source, target)] = \
            _lateralize(path, node_positions[source], node_positions[target], above)
    return edge_paths


def _lateralize(path, p1, p2, above):
    if not np.all(np.isclose(p1, p2)):
        gradient, intercept = _get_gradient_and_intercept(p1, p2)
    else:
        gradient, intercept = 0., p1[1]
    mask = _is_above_line(path, gradient, intercept)
    if above:
        mask = np.invert(mask)
    path[mask] = _reflect_across_line(path[mask], gradient, intercept)
    return path


class BaseArcDiagram(BaseGraph):
    """The arc diagram base class.

    Parameters
    ----------
    edges : list
        The edges of the graph, with each edge being represented by a (source node ID, target node ID) tuple.
    nodes : list or None, default None
        List of nodes. Required argument if any node in the graph is unconnected.
        If None, `nodes` is initialised to the set of the flattened `edges`.
    node_order : list or None, default None
        The ordering of nodes (left-to-right).
        If None, the node order is optimised such that the number of edge crossings is minimal.
    above : bool, default True
        If True, edges arc above the line of nodes.
        If False, edges arc below.
    node_layout : str or dict, default 'linear'
        If `node_layout` is a string, the node positions are computed using the indicated method:

        - 'random'    : place nodes in random positions;
        - 'circular'  : place nodes regularly spaced on a circle;
        - 'spring'    : place nodes using a force-directed layout (Fruchterman-Reingold algorithm);
        - 'dot'       : place nodes using the Sugiyama algorithm; the graph should be directed and acyclic;
        - 'community' : place nodes such that nodes belonging to the same community are grouped together
        - 'linear'    : place nodes one a horizontal line

        If `node_layout` is a dict, keys are nodes and values are (x, y) positions.
    node_layout_kwargs : dict or None, default None
        Keyword arguments passed to node layout functions.
        See the documentation of the following functions for a full description of available options:

        - get_random_layout
        - get_circular_layout
        - get_fruchterman_reingold_layout
        - get_sugiyama_layout
        - get_community_layout
        - get_linear_layout

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
        If the type is an int, all nodes have the same zorder.
        If the type is dict, maps each node to an individual zorder.
        If None, the edges will be plotted in the order they appear in 'adjacency'.
        Hint: graphs typically appear more visually pleasing if darker edges are plotted on top of lighter edges.
    arrows : bool, default False
        If True, draw edges with arrow heads.
    edge_layout_kwargs : dict, default None
        Keyword arguments passed to :code:`get_arced_edge_paths` and :code:`get_selfloop_paths`.
        Possible keyword arguments are:

        - :code:`rad` : float, default 1.
          Controls the curvature of the arcs.
        - :code:`selfloop_radius` : float, default 0.05 * np.linalg.norm(scale)
          Selfloop radius.
        - :code:`selfloop_angle` : float, default np.pi / 2
          Selfloop start angle

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
    BaseGraph, ArcDiagram, InteractiveArcDiagram, EditableArcDiagram

    """

    def __init__(self, edges, nodes=None, node_layout='linear', node_order=None, above=True, *args, **kwargs):
        self.above = above
        kwargs.setdefault('node_layout_kwargs', dict())
        kwargs['node_layout_kwargs'].setdefault('node_order', node_order)
        if node_order:
            kwargs['node_layout_kwargs'].setdefault('reduce_edge_crossings', False)
        else:
            kwargs['node_layout_kwargs'].setdefault('reduce_edge_crossings', True)
        super().__init__(edges, nodes=nodes, node_layout=node_layout, edge_layout='arc', *args, **kwargs)

    def _get_edge_paths(self, edges, node_positions, edge_layout, edge_layout_kwargs):
        edge_paths = super()._get_edge_paths(edges, node_positions, edge_layout, edge_layout_kwargs)
        edge_paths = _lateralize_arced_edge_paths(edge_paths, node_positions, self.above)
        return edge_paths


class ArcDiagram(BaseArcDiagram, Graph):
    """
    Parses the given graph data object and initialises the BaseArcDiagram object.

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

    node_order : list or None, default None
        The ordering of nodes (left-to-right).
        If None, the node order is optimised such that the number of edge crossings is minimal.
    above : bool, default True
        If True, edges arc above the line of nodes.
        If False, edges arc below.
    node_layout : str or dict, default 'linear'
        If `node_layout` is a string, the node positions are computed using the indicated method:

        - 'random'    : place nodes in random positions;
        - 'circular'  : place nodes regularly spaced on a circle;
        - 'spring'    : place nodes using a force-directed layout (Fruchterman-Reingold algorithm);
        - 'dot'       : place nodes using the Sugiyama algorithm; the graph should be directed and acyclic;
        - 'community' : place nodes such that nodes belonging to the same community are grouped together
        - 'linear'    : place nodes one a horizontal line

        If `node_layout` is a dict, keys are nodes and values are (x, y) positions.
    node_layout_kwargs : dict or None, default None
        Keyword arguments passed to node layout functions.
        See the documentation of the following functions for a full description of available options:

        - get_random_layout
        - get_circular_layout
        - get_fruchterman_reingold_layout
        - get_sugiyama_layout
        - get_community_layout
        - get_linear_layout

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
        If the type is an int, all nodes have the same zorder.
        If the type is dict, maps each node to an individual zorder.
        If None, the edges will be plotted in the order they appear in 'adjacency'.
        Hint: graphs typically appear more visually pleasing if darker edges are plotted on top of lighter edges.
    arrows : bool, default False
        If True, draw edges with arrow heads.
    edge_layout_kwargs : dict, default None
        Keyword arguments passed to :code:`get_arced_edge_paths` and :code:`get_selfloop_paths`.
        Possible keyword arguments are:

        - :code:`rad` : float, default 1.
          Controls the curvature of the arcs.
        - :code:`selfloop_radius` : float, default 0.05 * np.linalg.norm(scale)
          Selfloop radius.
        - :code:`selfloop_angle` : float, default np.pi / 2
          Selfloop start angle

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
    Graph, InteractiveArcDiagram, EditableArcDiagram

    """

    def __init__(self, graph, node_layout='linear', node_order=None, above=True, *args, **kwargs):
        self.above = above
        kwargs.setdefault('node_layout_kwargs', dict())
        kwargs['node_layout_kwargs'].setdefault('node_order', node_order)
        if node_order:
            kwargs['node_layout_kwargs'].setdefault('reduce_edge_crossings', False)
        else:
            kwargs['node_layout_kwargs'].setdefault('reduce_edge_crossings', True)
        Graph.__init__(self, graph, node_layout=node_layout, edge_layout='arc', *args, **kwargs)


class DraggableArcDiagram(ArcDiagram, DraggableGraph):

    def __init__(self, *args, **kwargs):
        ArcDiagram.__init__(self, *args, **kwargs)
        DraggableArtists.__init__(self, self.node_artists.values())

        self._draggable_artist_to_node = dict(zip(self.node_artists.values(), self.node_artists.keys()))
        self._clickable_artists.extend(list(self.edge_artists.values()))
        self._selectable_artists.extend(list(self.edge_artists.values()))
        self._base_linewidth.update(dict([(artist, artist._lw_data) for artist in self.edge_artists.values()]))
        self._base_edgecolor.update(dict([(artist, artist.get_edgecolor()) for artist in self.edge_artists.values()]))

        # # trigger resize of labels when canvas size changes
        # self.fig.canvas.mpl_connect('resize_event', self._on_resize)

    def _update_node_positions(self, nodes, cursor_position):
        # cursor_position[1] = 0. # remove y-component to remain on the line
        for node in nodes:
            x, _ = cursor_position + self._offset[self.node_artists[node]]
            self.node_positions[node] = np.array([x, self.node_positions[node][1]])

    def _update_edges(self, edges):
        edge_paths = dict()
        edge_paths.update(self._update_arced_edge_paths([(source, target) for (source, target) in edges if source != target]))
        edge_paths.update(self._update_selfloop_paths([(source, target) for (source, target) in edges if source == target]))
        edge_paths = _lateralize_arced_edge_paths(edge_paths, self.node_positions, self.above)
        self.edge_paths.update(edge_paths)
        self._update_edge_artists(edge_paths)


class InteractiveArcDiagram(DraggableArcDiagram, EmphasizeOnHoverGraph, AnnotateOnClickGraph, TableOnClickGraph):
    """Extends the `ArcDiagram` class to support node placement with the mouse, emphasis of graph elements when hovering over them, and toggleable annotations.

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

    node_order : list or None, default None
        The ordering of nodes (left-to-right).
        If None, the node order is optimised such that the number of edge crossings is minimal.
    above : bool, default True
        If True, edges arc above the line of nodes.
        If False, edges arc below.
    node_layout : str or dict, default 'linear'
        If `node_layout` is a string, the node positions are computed using the indicated method:

        - 'random'    : place nodes in random positions;
        - 'circular'  : place nodes regularly spaced on a circle;
        - 'spring'    : place nodes using a force-directed layout (Fruchterman-Reingold algorithm);
        - 'dot'       : place nodes using the Sugiyama algorithm; the graph should be directed and acyclic;
        - 'community' : place nodes such that nodes belonging to the same community are grouped together
        - 'linear'    : place nodes one a horizontal line

        If `node_layout` is a dict, keys are nodes and values are (x, y) positions.
    node_layout_kwargs : dict or None, default None
        Keyword arguments passed to node layout functions.
        See the documentation of the following functions for a full description of available options:

        - get_random_layout
        - get_circular_layout
        - get_fruchterman_reingold_layout
        - get_sugiyama_layout
        - get_community_layout
        - get_linear_layout

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
        If the type is an int, all nodes have the same zorder.
        If the type is dict, maps each node to an individual zorder.
        If None, the edges will be plotted in the order they appear in 'adjacency'.
        Hint: graphs typically appear more visually pleasing if darker edges are plotted on top of lighter edges.
    arrows : bool, default False
        If True, draw edges with arrow heads.
    edge_layout_kwargs : dict, default None
        Keyword arguments passed to :code:`get_arced_edge_paths` and :code:`get_selfloop_paths`.
        Possible keyword arguments are:

        - :code:`rad` : float, default 1.
          Controls the curvature of the arcs.
        - :code:`selfloop_radius` : float, default 0.05 * np.linalg.norm(scale)
          Selfloop radius.
        - :code:`selfloop_angle` : float, default np.pi / 2
          Selfloop start angle

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
    ArcDiagram, EditableArcDiagram

    Notes
    -----
    You must retain a reference to the plot instance!
    Otherwise, the plot instance will be garbage collected after the initial draw
    and you won't be able to move the plot elements around.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from netgraph import InteractiveArcDiagram
    >>> plt.ion()
    >>> plot_instance = InteractiveArcDiagram(my_graph_obj)
    >>> plt.show()

    """

    def __init__(self, *args, **kwargs):

        DraggableArcDiagram.__init__(self, *args, **kwargs)

        artists = list(self.node_artists.values()) + list(self.edge_artists.values())
        keys = list(self.node_artists.keys()) + list(self.edge_artists.keys())
        self.artist_to_key = dict(zip(artists, keys))
        EmphasizeOnHover.__init__(self, artists)
        self.mouseover_highlight_mapping = self._get_default_mouseover_highlight_mapping()

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

        if 'tables' in kwargs:
            artist_to_table = dict()
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
        DraggableArcDiagram._on_motion(self, event)
        EmphasizeOnHoverGraph._on_motion(self, event)


    def _on_release(self, event):
        if self._currently_dragging is False:
            DraggableArcDiagram._on_release(self, event)
            if self.artist_to_annotation:
                AnnotateOnClickGraph._on_release(self, event)
            if hasattr(self, 'artist_to_table'):
                TableOnClickGraph._on_release(self, event)
        else:
            DraggableArcDiagram._on_release(self, event)
            if self.artist_to_annotation:
                self._redraw_annotations(event)


    def _redraw_annotations(self, event):
        if event.inaxes == self.ax:
            for artist in self.annotated_artists:
                self._remove_annotation(artist)
                placement = self._get_annotation_placement(artist)
                self._add_annotation(artist, *placement)
            self.fig.canvas.draw()


class NascentEdge(plt.Line2D):

    def __init__(self, source, origin, rad=1., above=True):
        self.source = source
        self.origin = origin
        self.rad = rad
        self.above = above
        x, y = self._get_arc(self.origin, self.origin).T
        super().__init__(x, y, color='lightgray', linestyle='--')

    def _get_arc(self, p0, p1):
        arc_factory = ConnectionStyle.Arc3(rad=self.rad)
        path = arc_factory(p0, p1, shrinkA=0., shrinkB=0.)
        smoothed = _bspline(path.vertices, 100)
        return _lateralize(smoothed, p0, p1, self.above)

    def _update(self, x1, y1):
        x, y = self._get_arc(self.origin, (x1, y1)).T
        super().set_data(x, y)


class MutableArcDiagram(InteractiveArcDiagram, MutableGraph):
    """Extends `InteractiveArcDiagram` to support the addition or removal of nodes and edges.

    - Double clicking on two nodes successively will create an edge between them.
    - Pressing 'insert' or '+' will add a new node to the graph.
    - Pressing 'delete' or '-' will remove selected nodes and edges.
    - Pressing '@' will reverse the direction of selected edges.

    Notes
    -----
    When adding a new node, the properties of the last selected node will be used to style the node artist.
    Ditto for edges. If no node or edge has been previously selected the first created node or edge artist will be used.

    See also
    --------
    ArcDiagram, InteractiveArcDiagram

    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self._reverse_node_artists = {artist : node for node, artist in self.node_artists.items()}
        self._reverse_edge_artists = {artist : edge for edge, artist in self.edge_artists.items()}
        self._last_selected_node_properties = self._extract_node_properties(next(iter(self.node_artists.values())))
        self._last_selected_edge_properties = self._extract_edge_properties(next(iter(self.edge_artists.values())))
        self._nascent_edge = None

        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)


    def _on_press(self, event):
        InteractiveArcDiagram._on_press(self, event)

        if event.inaxes == self.ax:
            for artist in self._clickable_artists:
                if artist.contains(event)[0]:
                    self._extract_artist_properties(artist)
                    break

            if event.dblclick:
                self._add_or_remove_nascent_edge(event)


    def _add_nascent_edge(self, node):
        nascent_edge = NascentEdge(node, self.node_positions[node], above=self.above)
        self.ax.add_artist(nascent_edge)
        return nascent_edge


    def _on_motion(self, event):
        super()._on_motion(event)

        if event.inaxes == self.ax:
            if self._nascent_edge:
                self._nascent_edge._update(event.xdata, event.ydata)
                self.fig.canvas.draw_idle()


    def _set_position_of_newly_created_node(self, x, y):
        node_positions = list(self.node_positions.values())
        if _are_collinear(node_positions):
            line = node_positions[:2]
            x, y = _get_orthogonal_projection_onto_segment((x, y), line)
        return (x, y)


class EditableArcDiagram(MutableArcDiagram, EditableGraph):
    """Extends `InteractiveArcDiagram` to support adding, deleting, and editing graph elements interactively.

    a) Addition and removal of nodes and edges:

    - Double clicking on two nodes successively will create an edge between them.
    - Pressing 'insert' or '+' will add a new node to the graph.
    - Pressing 'delete' or '-' will remove selected nodes and edges.
    - Pressing '@' will reverse the direction of selected edges.

    b) Creation and editing of labels and annotations:

    - To create or edit a node or edge label, select the node (or edge) artist, press the 'enter' key, and type.
    - To create or edit an annotation, select the node (or edge) artist, press 'alt'+'enter', and type.
    - Terminate either action by pressing 'enter' or 'alt'+'enter' a second time.

    Notes
    -----
    When adding a new node, the properties of the last selected node will be used to style the node artist.
    Ditto for edges. If no node or edge has been previously selected the first created node or edge artist will be used.

    See also
    --------
    ArcDiagram, InteractiveArcDiagram

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initiate node and edge label data structures if they don't exist
        if not hasattr(self, 'node_label_artists'):
            node_labels = {node : '' for node in self.nodes}
            self.node_label_fontdict = self._initialize_node_label_fontdict(
                kwargs.get('node_label_fontdict'), node_labels, kwargs.get('node_label_offset', (0., 0.)))
            self.node_label_offset, self._recompute_node_label_offsets =\
                self._initialize_node_label_offset(node_labels, kwargs.get('node_label_offset', (0., 0.)))
            if self._recompute_node_label_offsets:
                self._update_node_label_offsets()
            self.node_label_artists = dict()
            self.draw_node_labels(node_labels, self.node_label_fontdict)

        if not hasattr(self, 'edge_label_artists'):
            edge_labels = {edge : '' for edge in self.edges}
            self.edge_label_fontdict = self._initialize_edge_label_fontdict(kwargs.get('edge_label_fontdict'))
            self.edge_label_position = kwargs.get('edge_label_position', 0.5)
            self.edge_label_rotate = kwargs.get('edge_label_rotate', True)
            self.edge_label_artists = dict()
            self.draw_edge_labels(edge_labels, self.edge_label_position,
                                  self.edge_label_rotate, self.edge_label_fontdict)

        self._currently_writing_labels = False
        self._currently_writing_annotations = False
