#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Implements the interactive multi-graph classes, specifically

  - InteractiveMultiGraph,
  - MutableMultiGraph,
  - EditableMultiGraph,

as well as the following helper classes:

  - DraggableMultiGraph,
  - DraggableMultiGraphWithGridMode,
  - EmphasizeOnKeyPress,
  - CycleEmphasisOnKeyPress, and
  - CycleEmphasisOnKeyPressGraph.

"""

import warnings
import numpy as np
import matplotlib.pyplot as plt

from ._graph_classes import Graph
from ._interactive_graph_classes import (
    DraggableArtists,
    DraggableGraph,
    DraggableGraphWithGridMode,
    EmphasizeOnHoverGraph,
    AnnotateOnClickGraph,
    TableOnClickGraph,
    MutableGraph,
    EditableGraph,
)
from ._multigraph_classes import MultiGraph
from ._utils import _get_unique_nodes, _save_cast_str_to_int


# from: https://matplotlib.org/stable/tutorials/introductory/customizing.html#the-default-matplotlibrc-file
MATPLOTLIB_RESERVED_KEYS = [
    "f",
    "h", "r", "home",
    "left", "c", "backspace",
    "right", "v",
    "p",
    "o",
    "s",
    "q",
    "g", "G",
    "l",
    "k", "L",
]


class DraggableMultiGraph(MultiGraph, DraggableGraph, DraggableArtists):
    """Augments :py:class:`MultiGraph` to support selection and dragging of node artists with the mouse.

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

    *args, **kwargs
        Parameters passed through to :py:class:`Graph`.
        See its documentation for a full list of available arguments.

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
    :py:class:`MultiGraph`, :py:class:`DraggableGraph`, :py:class:`DraggableArtists`, :py:class:`InteractiveMultiGraph`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        DraggableArtists.__init__(self, self.node_artists.values())
        self._setup_dragging_clicking_and_selecting()


class DraggableMultiGraphWithGridMode(DraggableMultiGraph, DraggableGraphWithGridMode):
    """
    Adds a grid-mode to :py:class:`DraggableMultiGraph`, in which node positions are fixed to a grid.
    To activate, press the letter 'g'.

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

    *args, **kwargs
        Parameters passed through to :py:class:`MultiGraph`.
        See its documentation for a full list of available arguments.

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
    :py:class:`DraggableMultiGraph`, :py:class:`DraggableGraphWithGridMode`, :py:class:`InteractiveMultiGraph`

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_grid_mode()


class EmphasizeOnKeyPress(object):
    """Emphasize groups of matplotlib artists by pressing the corresponding key.
    Pressing escape removes any emphasis.

    """

    def __init__(self, artists, emphasis_group_to_artists):

        self.emphasizeable_artists = artists
        self._emphasis_group_to_artists = emphasis_group_to_artists
        self._base_alpha = {artist : artist.get_alpha() for artist in self.emphasizeable_artists}
        self.deemphasized_artists = []

        conflicts = [group for group in self._emphasis_group_to_artists if group in MATPLOTLIB_RESERVED_KEYS]
        if conflicts:
            msg = f"The following emphasis group keys are reserved by Matplotlib: {conflicts}"
            msg += "\nPressing these keys to emphasize the corresponding artists may have unintended side-effects."
            warnings.warn(msg)

        if not hasattr(self, "fig"):
            try:
                self.fig, = set(list(artist.figure for artist in artists))
            except ValueError:
                raise Exception("All artists have to be on the same figure!")

        if not hasattr(self, "ax"):
            try:
                self.ax, = set(list(artist.axes for artist in artists))
            except ValueError:
                raise Exception("All artists have to be on the same axis!")

        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)


    def _on_key_press(self, event):
        group = _save_cast_str_to_int(event.key)
        if group in self._emphasis_group_to_artists:
            self._emphasize_group(group)
        elif group == "escape":
            self._remove_emphasis()


    def _emphasize_group(self, group):
        for artist in self.emphasizeable_artists:
            if (artist in self._emphasis_group_to_artists[group]):
                if artist in self.deemphasized_artists:
                    artist.set_alpha(self._base_alpha[artist])
                    self.deemphasized_artists.remove(artist)
            else:
                if (artist not in self.deemphasized_artists):
                    artist.set_alpha(self._base_alpha[artist]/5)
                    self.deemphasized_artists.append(artist)
        self.fig.canvas.draw_idle()


    def _remove_emphasis(self):
        for artist in self.deemphasized_artists:
            artist.set_alpha(self._base_alpha[artist])
        self.deemphasized_artists = []
        self.fig.canvas.draw_idle()


class CycleEmphasisOnKeyPress(EmphasizeOnKeyPress):
    """Emphasize groups of matplotlib artists by pressing the
    corresponding key. Alternatively, the 'up' and 'down' keys can be
    used to cycle through the groups. Pressing escape removes any
    emphasis.

    """

    def __init__(self, artists, emphasis_group_to_artists):
        super().__init__(artists, emphasis_group_to_artists)
        self._current_emphasis_group = list(emphasis_group_to_artists.keys())[-1]
        self._index_to_emphasis_group = dict(enumerate(emphasis_group_to_artists))
        self._emphasis_group_to_index = {v : k for k, v in self._index_to_emphasis_group.items()}


    def _on_key_press(self, event):
        group = _save_cast_str_to_int(event.key)
        if group in self._emphasis_group_to_artists:
            self._emphasize_group(group)
        elif group == "up":
            self._emphasize_group(self._cycle_emphasis_group(1))
        elif group == "down":
            self._emphasize_group(self._cycle_emphasis_group(-1))
        elif group == "escape":
            self._remove_emphasis()


    def _emphasize_group(self, group):
        super()._emphasize_group(group)
        self._current_emphasis_group = group


    def _cycle_emphasis_group(self, step):
        last_idx = self._emphasis_group_to_index[self._current_emphasis_group]
        next_idx = (last_idx + step) % len(self._emphasis_group_to_artists)
        return self._index_to_emphasis_group[next_idx]


class CycleEmphasisOnKeyPressGraph(Graph, CycleEmphasisOnKeyPress):

    def __init__(self, *args, **kwargs):
        Graph.__init__(self, *args, **kwargs)
        self._setup_cycle_emphasis_on_key_press(*args, **kwargs)


    def _setup_cycle_emphasis_on_key_press(self, *args, **kwargs):
        if "emphasis_groups" in kwargs:
            emphasis_groups = kwargs["emphasis_groups"]
            emphasis_group_to_artists = dict()
            for group, items in emphasis_groups.items():
                artists = []
                for item in items:
                    if item in self.nodes:
                        artists.append(self.node_artists[item])
                    elif item in self.edges:
                        artists.append(self.edge_artists[item])
                    else:
                        warnings.warn(f"'{item}' neither a valid node nor edge identifier.")
                emphasis_group_to_artists[group] = artists
            artists = list(self.node_artists.values()) + list(self.edge_artists.values())
            CycleEmphasisOnKeyPress.__init__(self, artists, emphasis_group_to_artists)


class InteractiveMultiGraph(DraggableMultiGraphWithGridMode, EmphasizeOnHoverGraph, CycleEmphasisOnKeyPressGraph, AnnotateOnClickGraph, TableOnClickGraph):
    """Extends the :py:class:`MultiGraph` class to support node placement with
    the mouse, emphasis of graph elements, and toggleable annotations.

    - Nodes can be selected and dragged around with the mouse.
    - Nodes and edges are emphasized when hovering over them.
    - If subsets of edges share the same edge key, and the edge key is an integer or a single letter, pressing that key emphasizes all corresponding edges. The up and down keys can be used to cycle through the different subsets. Pressing escape removes all emphasis.
    - Supports additional annotations that can be toggled on and off by clicking on the corresponding node or edge.
    - These annotations can also be tables.

    Parameters
    ----------
    graph : various formats
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
    annotations : dict
        Mapping of nodes or edges to strings or dictionaries, the annotations.
        The visibility of the annotations can be toggled on or off by clicking on the corresponding node or edge.

        .. code-block::

           annotations = {
               0         : 'Normal node',
               1         : {s : 'Less important node', fontsize : 2},
               2         : {s : 'Very important node', fontcolor : 'red'},
               (0, 1, 0) : 'Normal edge',
               (1, 2, 0) : {s : 'Less important edge', fontsize : 2},
               (2, 0, 1) : {s : 'Very important edge', fontcolor : 'red'},
           }

    annotation_fontdict : dict
        Keyword arguments passed to matplotlib.text.Text if only the annotation string is given.
        For a full list of available arguments see the matplotlib documentation.
    tables : dict node/edge : pandas dataframe
        Mapping of nodes and/or edges to pandas dataframes.
        The visibility of the tables that can toggled on or off by clicking on the corresponding node or edge.
    table_kwargs : dict
        Keyword arguments passed to matplotlib.pyplot.table.

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
    :py:class:`MultiGraph`, :py:class:`InteractiveGraph`, :py:class:`EditableMultiGraph`

    Notes
    -----
    You must retain a reference to the plot instance!
    Otherwise, the plot instance will be garbage collected after the initial draw
    and you won't be able to move the plot elements around.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from netgraph import InteractiveMultiGraph
    >>> plt.ion()
    >>> plot_instance = InteractiveMultiGraph(my_graph_obj)
    >>> plt.show()

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_emphasis()
        self._setup_cycle_emphasis_on_key_press(*args, **kwargs)
        self._setup_annotations(*args, **kwargs)
        self._setup_table_annotations(*args, **kwargs)


    def _setup_cycle_emphasis_on_key_press(self, *args, **kwargs):
        emphasis_groups = dict()
        for edge in self.edges:
            key = edge[2] # i.e. the edge id
            if key in emphasis_groups:
                emphasis_groups[key].append(edge)
            else:
                emphasis_groups[key] = [edge]
        for key, edges in emphasis_groups.items():
            nodes = _get_unique_nodes(edges)
            emphasis_groups[key] = nodes + edges
        kwargs["emphasis_groups"] = emphasis_groups
        super()._setup_cycle_emphasis_on_key_press(*args, **kwargs)


    def _on_motion(self, event):
        DraggableMultiGraphWithGridMode._on_motion(self, event)
        EmphasizeOnHoverGraph._on_motion(self, event)


    def _on_release(self, event):
        if self._currently_dragging is False:
            DraggableMultiGraphWithGridMode._on_release(self, event)
            if self.artist_to_annotation:
                AnnotateOnClickGraph._on_release(self, event)
            if self.artist_to_table:
                TableOnClickGraph._on_release(self, event)
        else:
            DraggableMultiGraphWithGridMode._on_release(self, event)
            if self.artist_to_annotation:
                self._redraw_annotations(event)


class MutableMultiGraph(InteractiveMultiGraph, MutableGraph):
    """Extends :py:class:`InteractiveMultiGraph` to support the addition or removal of nodes and edges.

    - Double clicking on two nodes successively will create an edge between them.
    - Pressing 'insert' or '+' will add a new node to the graph.
    - Pressing 'delete' or '-' will remove selected nodes and edges.
    - Pressing '@' will reverse the direction of selected edges.

    When adding a new node, the properties of the last selected node will be used to style the node artist.
    This also applies to adding new edges, which additionally inherit the edge key from the last selected edge.
    If no node or edge has been previously selected the first created node or edge artist will be used instead.

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

    *args, **kwargs
        Parameters passed through to :py:class:`InteractiveMultiGraph`.
        See its documentation for a full list of available arguments.

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
    :py:class:`InteractiveMultiGraph`, :py:class:`MutableGraph`, :py:class:`EditableMultiGraph`

    """

    def __init__(self, graph, *args, **kwargs):

        if is_order_zero(graph):
            # The graph is order-zero, i.e. it has no edges and no nodes.
            # We hence initialise with a single edge, which populates
            # - last_selected_node_properties
            # - last_selected_edge_properties
            # with the chosen parameters.
            # We then delete the edge and the two nodes and return the empty canvas.
            source, target = DEFAULT_EDGE
            edge = (source, target, DEFAULT_EDGE_KEY)
            super().__init__([edge], *args, **kwargs)
            self._initialize_data_structures()
            self._delete_edge(edge)
            self._delete_node(source)
            self._delete_node(target)

        elif is_empty(graph):
            # The graph is empty, i.e. it has at least one node but no edges.
            nodes, _, _ = parse_graph(graph)
            if len(nodes) > 1:
                edge = (nodes[0], nodes[1], DEFAULT_EDGE_KEY)
                super().__init__([edge], nodes=nodes, *args, **kwargs)
                self._initialize_data_structures()
                self._delete_edge(edge)
            else: # single node
                node = nodes[0]
                dummy = 0 if node != 0 else 1
                edge = (node, dummy, DEFAULT_EDGE_KEY)
                super().__init__([edge], *args, **kwargs)
                self._initialize_data_structures()
                self._delete_edge(edge)
                self._delete_node(dummy)
        else:
            super().__init__(graph, *args, **kwargs)
            self._initialize_data_structures()

        # Ignore data limits and return full canvas.
        xmin, ymin = self.origin
        dx, dy = self.scale
        self.ax.axis([xmin, xmin+dx, ymin, ymin+dy])
        if self.autoscale_node_labels:
            self._rescale_node_labels()
        self.fig.canvas.draw()

        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)


    def _initialize_data_structures(self):
        super()._initialize_data_structures()
        self._last_selected_edge_key = self.edges[0][2]


    def _extract_artist_properties(self, artist):
        super()._extract_artist_properties(artist)
        if isinstance(artist, EdgeArtist):
            self._last_selected_edge_key = self._reverse_edge_artists[artist][2]


    def _on_motion(self, event):
        super()._on_motion(event)

        if event.inaxes == self.ax:
            if self._nascent_edge:
                self._nascent_edge._update(event.xdata, event.ydata)
                self.fig.canvas.draw_idle()


    def _add_or_remove_nascent_edge(self, event):
        for node, artist in self.node_artists.items():
            if artist.contains(event)[0]:
                if self._nascent_edge:
                    # connect edge to target node
                    edge = (self._nascent_edge.source, node, self._last_selected_edge_key)
                    if edge not in self.edges:
                        self._add_edge(*edge)
                        self.edge_layout.get()
                        self._update_edge_artists()
                    else:
                        print(f"Edge already exists: {edge}")
                    self._remove_nascent_edge()
                else:
                    self._nascent_edge = self._add_nascent_edge(node)
                break
        else:
            if self._nascent_edge:
                self._remove_nascent_edge()


    def _on_key_press(self, event):
        InteractiveMultiGraph._on_key_press(self, event)
        MutableGraph._on_key_press(self, event)


    def _add_edge(self, source, target, key=None, edge_properties=None):

        if key is None:
            key = self._last_selected_edge_key
        edge = (source, target, key)

        if not edge_properties:
            edge_properties = self._last_selected_edge_properties

        # path = np.array([self.node_positions[source], self.node_positions[target]])
        self.edge_layout.add_edge(edge, edge_properties["width"])
        if source != target:
            edge_paths = self.edge_layout.approximate_nonloop_edge_paths([(source, target)])
        else:
            edge_paths = self.edge_layout.approximate_selfloop_edge_paths([(source, target)])
        path = edge_paths[edge]

        # create artist
        artist = EdgeArtist(midline=path, shape="full", **edge_properties)
        self.ax.add_patch(artist)

        # bookkeeping
        self._expand_edge_data_structures(edge, artist, path)

        return edge


    def _delete_edge(self, edge):
        artist = self.edge_artists[edge]
        self._contract_edge_data_structures(edge, artist)
        artist.remove()
        self.edge_layout.delete_edge(edge)


    def _reverse_edges(self):
        edges = [self._reverse_edge_artists[artist] for artist in self._selected_artists if isinstance(artist, EdgeArtist)]
        edge_properties = [self._extract_edge_properties(self.edge_artists[edge]) for edge in edges]

        # delete old edges;
        # note this step has to be completed before creating new edges,
        # as bi-directional edges can pose a problem otherwise
        for edge in edges:
            self._delete_edge(edge)

        for edge, properties in zip(edges, edge_properties):
            self._add_edge(edge[1], edge[0], edge[2], properties)

        self.edge_layout.get()
        self._update_edge_artists()


class EditableMultiGraph(MutableMultiGraph, EditableGraph):
    """Extends :py:class:`InteractiveMultiGraph` to support adding, deleting, and editing graph elements interactively.

    a) Addition and removal of nodes and edges:

    - Double clicking on two nodes successively will create an edge between them.
    - Pressing 'insert' or '+' will add a new node to the graph.
    - Pressing 'delete' or '-' will remove selected nodes and edges.
    - Pressing '@' will reverse the direction of selected edges.

    b) Creation and editing of labels and annotations:

    - To create or edit a node or edge label, select the node (or edge) artist, press the 'enter' key, and type.
    - To create or edit an annotation, select the node (or edge) artist, press 'alt'+'enter', and type.
    - Terminate either action by pressing 'enter' or 'alt'+'enter' a second time.

    When adding a new node, the properties of the last selected node will be used to style the node artist.
    Ditto for edges. If no node or edge has been previously selected the first created node or edge artist will be used.

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

    *args, **kwargs
        Parameters passed through to :py:class:`InteractiveMultiGraph`.
        See its documentation for a full list of available arguments.

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
    :py:class:`InteractiveMultiGraph`, :py:class:`EditableGraph`

    """

    def __init__(self, *args, **kwargs):
        MutableMultiGraph.__init__(self, *args, **kwargs)

        if not hasattr(self, 'node_label_artists'):
            self._initialize_empty_node_labels(kwargs)

        if not hasattr(self, 'edge_label_artists'):
            self._initialize_empty_edge_labels(kwargs)

        self._currently_writing_labels = False
        self._currently_writing_annotations = False

        # restore deprecated self.fig.canvas.manager.key_press method
        # https://github.com/matplotlib/matplotlib/issues/26713#issuecomment-1709938648
        self.fig.canvas.manager.key_press = key_press_handler


    def _on_key_press(self, event):
        MutableMultiGraph._on_key_press(self, event)
        EditableGraph._on_key_press(self, event)
