#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
InteractiveGraph variants.
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt

from functools import partial
from matplotlib.patches import Rectangle
from matplotlib.backend_bases import key_press_handler

try:
    from ._main import InteractiveGraph, BASE_SCALE, DraggableGraph
    from ._line_supercover import line_supercover
    from ._artists import NodeArtist, EdgeArtist
    from ._parser import is_order_zero, is_empty, parse_graph
    from ._multigraph import InteractiveMultiGraph
    from ._artists import NodeArtist, CircularNodeArtist, RegularPolygonNodeArtist
except ValueError:
    from _main import InteractiveGraph, BASE_SCALE
    from _line_supercover import line_supercover
    from _parser import is_order_zero, is_empty, parse_graph
    from _multigraph import InteractiveMultiGraph
    from _artists import NodeArtist, CircularNodeArtist, RegularPolygonNodeArtist


DEFAULT_EDGE = (0, 1)
DEFAULT_EDGE_KEY = 0


class NascentEdge(plt.Line2D):
    def __init__(self, source, origin):
        self.source = source
        self.origin = origin
        x0, y0 = origin
        super().__init__([x0, x0], [y0, y0], color='lightgray', linestyle='--')

    def _update(self, x1, y1):
        x0, y0 = self.origin
        super().set_data([[x0, x1], [y0, y1]])


class MutableGraph(InteractiveGraph):
    """Extends :py:class:`InteractiveGraph` to support the addition or removal of nodes and edges.

    - Double clicking on two nodes successively will create an edge between them.
    - Pressing 'insert' or '+' will add a new node to the graph.
    - Pressing 'delete' or '-' will remove selected nodes and edges.
    - Pressing '@' will reverse the direction of selected edges.

    When adding a new node, the properties of the last selected node will be used to style the node artist.
    Ditto for edges. If no node or edge has been previously selected the first created node or edge artist will be used.

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

        - :code:`networkx.Graph`, :code:`igraph.Graph`, or :code:`graph_tool.Graph` object

    *args, **kwargs
        Parameters passed through to :py:class:`InteractiveGraph`.
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
    :py:class:`InteractiveGraph`, :py:class:`EditableGraph`, :py:class:`MutableMultiGraph`

    """

    def __init__(self, graph, *args, **kwargs):

        if is_order_zero(graph):
            # The graph is order-zero, i.e. it has no edges and no nodes.
            # We hence initialise with a single edge, which populates
            # - last_selected_node_properties
            # - last_selected_edge_properties
            # with the chosen parameters.
            # We then delete the edge and the two nodes and return the empty canvas.
            super().__init__([(0, 1)], *args, **kwargs)
            self._initialize_data_structures()
            self._delete_edge((0, 1))
            self._delete_node(0)
            self._delete_node(1)

        elif is_empty(graph):
            # The graph is empty, i.e. it has at least one node but no edges.
            nodes, _, _ = parse_graph(graph)
            if len(nodes) > 1:
                edge = (nodes[0], nodes[1])
                super().__init__([edge], nodes=nodes, *args, **kwargs)
                self._initialize_data_structures()
                self._delete_edge(edge)
            else: # single node
                node = nodes[0]
                dummy = 0 if node != 0 else 1
                edge = (node, dummy)
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
        self._rescale_node_labels()
        self.fig.canvas.draw()

        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)


    def _initialize_data_structures(self):
        self._reverse_node_artists = {artist : node for node, artist in self.node_artists.items()}
        self._reverse_edge_artists = {artist : edge for edge, artist in self.edge_artists.items()}
        self._last_selected_node_properties = self._extract_node_properties(next(iter(self.node_artists.values())))
        self._last_selected_node_type = type(next(iter(self.node_artists.values())))
        self._last_selected_edge_properties = self._extract_edge_properties(next(iter(self.edge_artists.values())))
        self._nascent_edge = None


    def _on_key_press(self, event):
        if event.key in ('insert', '+'):
            self._add_node(event)
        elif event.key in ('delete', '-'):
            self._delete_nodes()
            self._delete_edges()
        elif event.key == '@':
            self._reverse_edges()
        else:
            pass

        self.fig.canvas.draw_idle()


    def _on_press(self, event):
        # TODO : trigger this code on any node or edge selection;
        # clicking on a node or edge is just one of the ways to select them
        super()._on_press(event)

        if event.inaxes == self.ax:
            for artist in self._clickable_artists:
                if artist.contains(event)[0]:
                    self._extract_artist_properties(artist)
                    break

            if event.dblclick:
                self._add_or_remove_nascent_edge(event)
                self.fig.canvas.draw_idle()


    def _add_or_remove_nascent_edge(self, event):
        for node, artist in self.node_artists.items():
            if artist.contains(event)[0]:
                if self._nascent_edge:
                    # connect edge to target node
                    if (self._nascent_edge.source, node) not in self.edges:
                        self._add_edge(self._nascent_edge.source, node)
                        self.edge_layout.get()
                        self._update_edge_artists()
                    else:
                        print("Edge already exists!")
                    self._remove_nascent_edge()
                else:
                    self._nascent_edge = self._add_nascent_edge(node)
                break
        else:
            if self._nascent_edge:
                self._remove_nascent_edge()


    def _add_nascent_edge(self, node):
        nascent_edge = NascentEdge(node, self.node_positions[node])
        self.ax.add_artist(nascent_edge)
        return nascent_edge


    def _remove_nascent_edge(self):
        self._nascent_edge.remove()
        self._nascent_edge = None


    def _extract_artist_properties(self, artist):
        if isinstance(artist, NodeArtist):
            self._last_selected_node_properties = self._extract_node_properties(artist)
            self._last_selected_node_type = type(artist)
        elif isinstance(artist, EdgeArtist):
            self._last_selected_edge_properties = self._extract_edge_properties(artist)


    def _extract_node_properties(self, node_artist):
        properties = dict(
            size                 = node_artist.size,
            linewidth            = self._base_linewidth[node_artist],
            facecolor            = node_artist.get_facecolor(),
            edgecolor            = self._base_edgecolor[node_artist],
            alpha                = self._base_alpha[node_artist],
            zorder               = node_artist.get_zorder(),
        )
        if isinstance(node_artist, CircularNodeArtist):
            pass
        elif isinstance(node_artist, RegularPolygonNodeArtist):
            properties["orientation"] = node_artist.orientation
            properties["total_vertices"] = len(node_artist._path) - 1
        else: # NodeArtist
            properties["path"] = node_artist._path
            properties["orientation"] = node_artist.orientation
            properties["linewidth_correction"] = node_artist.linewidth_correction
        return properties


    def _extract_edge_properties(self, edge_artist):
        return dict(
            width       = edge_artist.width,
            facecolor   = edge_artist.get_facecolor(),
            alpha       = self._base_alpha[edge_artist],
            head_length = edge_artist.head_length,
            head_width  = edge_artist.head_width,
            edgecolor   = self._base_edgecolor[edge_artist],
            linewidth   = self._base_linewidth[edge_artist],
            head_offset = edge_artist.head_offset, # TODO: change to target node size
            tail_offset = edge_artist.tail_offset, # TODO: change to source node size
            curved      = edge_artist.curved,
            zorder      = edge_artist.get_zorder(),
        )


    def _on_motion(self, event):
        super()._on_motion(event)

        if event.inaxes == self.ax:
            if self._nascent_edge:
                self._nascent_edge._update(event.xdata, event.ydata)
                self.fig.canvas.draw_idle()


    def _add_node(self, event):
        if event.inaxes != self.ax:
            print('Position outside of axis limits! Cannot create node.')
            return

        # create node ID; use smallest unused int
        node = len(self.nodes)
        while node in self.node_positions.keys():
            node += 1

        # get position of cursor place node at cursor position
        pos = self._set_position_of_newly_created_node(event.xdata, event.ydata)

        # copy attributes of last selected artist;
        # if none is selected, use a random artist
        if self._selected_artists:
            node_properties = self._extract_node_properties(self._selected_artists[-1])
            node_type = type(self._selected_artists[-1])
        else:
            node_properties = self._last_selected_node_properties
            node_type = self._last_selected_node_type

        artist = node_type(xy=pos, **node_properties)
        self.ax.add_patch(artist)
        self._expand_node_data_structures(node, artist, pos)


    def _expand_node_data_structures(self, node, artist, pos):
        # 0) MutableGraph
        self._reverse_node_artists[artist] = node
        # 1) InteractiveGraph
        # 2a) DraggableGraph
        self._draggable_artist_to_node[artist] = node
        # 2b) EmphasizeOnHoverGraph
        self.artist_to_key[artist] = node
        # 2c) AnnotateOnClickGraph
        # None
        # 3a) Graph
        # None
        # 3b) ClickableArtists, SelectableArtists, DraggableArtists
        self._clickable_artists.append(artist)
        self._selectable_artists.append(artist)
        self._draggable_artists.append(artist)
        self._base_linewidth[artist] = artist._lw_data
        self._base_edgecolor[artist] = artist.get_edgecolor()
        # 3c) EmphasizeOnHover
        self.emphasizeable_artists.append(artist)
        self._base_alpha[artist] = artist.get_alpha()
        # 3d) AnnotateOnClick
        # None
        # 4) BaseGraph
        self.nodes.append(node)
        self.node_positions[node] = pos
        self.node_artists[node] = artist
        # self.node_label_artists # TODO (potentially)
        # self.node_label_offset  # TODO (potentially)
        # 5) edge layout engine
        self.edge_layout.add_node(node, pos)


    def _set_position_of_newly_created_node(self, x, y):
        return (x, y)


    def _delete_nodes(self):
        # translate selected artists into nodes
        nodes = [self._reverse_node_artists[artist] for artist in self._selected_artists if isinstance(artist, NodeArtist)]

        # delete edges to and from selected nodes
        edges = [edge for edge in self.edges if ((edge[0] in nodes) or (edge[1] in nodes))]
        for edge in edges:
            self._delete_edge(edge)

        # delete nodes
        for node in nodes:
            self._delete_node(node)


    def _delete_node(self, node):
        # print(f"Deleting node {node}.")
        artist = self.node_artists[node]
        self._contract_node_data_structures(node, artist)
        artist.remove()


    def _contract_node_data_structures(self, node, artist):
        # 0) MutableGraph
        del self._reverse_node_artists[artist]
        # 1) InteractiveGraph
        # None
        # 2a) DraggableGraph
        del self._draggable_artist_to_node[artist]
        # 2b) EmphasizeOnHoverGraph
        del self.artist_to_key[artist]
        # None
        # 2c) AnnotateOnClickGraph
        if artist in self.annotated_artists:
            self._remove_annotation(artist)
        # 3a) Graph
        # None
        # 3b) ClickableArtists, SelectableArtists, DraggableArtists
        self._clickable_artists.remove(artist)
        self._selectable_artists.remove(artist)
        self._draggable_artists.remove(artist)
        if artist in self._selected_artists:
            self._selected_artists.remove(artist)
        del self._base_linewidth[artist]
        del self._base_edgecolor[artist]
        # 3c) EmphasizeOnHover
        self.emphasizeable_artists.remove(artist)
        if artist in self.deemphasized_artists:
            self.deemphasized_artists.remove(artist)
        del self._base_alpha[artist]
        # 3d) AnnotateOnClick
        if artist in self.artist_to_annotation:
            del self.artist_to_annotation[artist]
        # 4) BaseGraph
        self.nodes.remove(node)
        del self.node_positions[node]
        del self.node_artists[node]
        if hasattr(self, 'node_label_artists'):
            if node in self.node_label_artists:
                self.node_label_artists[node].remove()
                del self.node_label_artists[node]
        if hasattr(self, 'node_label_offset'):
            if node in self.node_label_offset:
                del self.node_label_offset[node]
        self.edge_layout.delete_node(node)



    def _add_edge(self, source, target, edge_properties=None):

        edge = (source, target)

        if not edge_properties:
            edge_properties = self._last_selected_edge_properties

        self.edge_layout.add_edge(edge)
        if source != target:
            edge_paths = self.edge_layout.approximate_nonloop_edge_paths([edge])
        else:
            edge_paths = self.edge_layout.approximate_selfloop_edge_paths([edge])
        path = edge_paths[edge]

        # create artist
        if (target, source) in self.edges:
            shape = 'right'
            self.edge_artists[(target, source)].shape = 'right'
            self.edge_artists[(target, source)]._update_path()
        else:
            shape = 'full'

        # create artist
        artist = EdgeArtist(midline=path, shape="full", **edge_properties)
        self.ax.add_patch(artist)

        # bookkeeping
        self._expand_edge_data_structures(edge, artist, path)


    def _expand_edge_data_structures(self, edge, artist, path):
        # 0) MutableGraph
        self._reverse_edge_artists[artist] = edge
        # 1) InteractiveGraph
        # 2a) DraggableGraph
        # None
        # 2b) EmphasizeOnHoverGraph
        self.artist_to_key[artist] = edge
        # 2c) AnnotateOnClickGraph
        # None
        # 3a) Graph
        # None
        # 3b) ClickableArtists, SelectableArtists, DraggableArtists
        self._clickable_artists.append(artist)
        self._selectable_artists.append(artist)
        self._base_linewidth[artist] = artist._lw_data
        self._base_edgecolor[artist] = artist.get_edgecolor()
        # 3c) EmphasizeOnHover
        self.emphasizeable_artists.append(artist)
        self._base_alpha[artist] = artist.get_alpha()
        # 3d) AnnotateOnClick
        # None
        # 4) BaseGraph
        self.edges.append(edge)
        self.edge_artists[edge] = artist


    def _delete_edges(self):
        edges = [self._reverse_edge_artists[artist] for artist in self._selected_artists if isinstance(artist, EdgeArtist)]
        for edge in edges:
            self._delete_edge(edge)


    def _delete_edge(self, edge):
        source, target = edge
        if (target, source) in self.edges:
            self.edge_artists[(target, source)].shape = 'full'
            self.edge_artists[(target, source)]._update_path()

        artist = self.edge_artists[edge]
        self._contract_edge_data_structures(edge, artist)
        artist.remove()
        self.edge_layout.delete_edge(edge)


    def _contract_edge_data_structures(self, edge, artist):
        # 0) MutableGraph
        del self._reverse_edge_artists[artist]
        # 1) InteractiveGraph
        # None
        # 2a) DraggableGraph
        # None
        # 2b) EmphasizeOnHoverGraph
        del self.artist_to_key[artist]
        # 2c) AnnotateOnClickGraph
        if artist in self.annotated_artists:
            self._remove_annotation(artist)
        # 3a) Graph
        # None
        # 3b) ClickableArtists, SelectableArtists, DraggableArtists
        self._clickable_artists.remove(artist)
        self._selectable_artists.remove(artist)
        try:
            self._selected_artists.remove(artist)
        except ValueError:
            pass
        del self._base_linewidth[artist]
        del self._base_edgecolor[artist]
        # 3c) EmphasizeOnHover
        self.emphasizeable_artists.remove(artist)
        if artist in self.deemphasized_artists:
            self.deemphasized_artists.remove(artist)
        del self._base_alpha[artist]
        # 3d) AnnotateOnClick
        if artist in self.artist_to_annotation:
            del self.artist_to_annotation[artist]
        # 4) BaseGraph
        self.edges.remove(edge)
        del self.edge_artists[edge]
        if hasattr(self, 'edge_label_artists'):
            if edge in self.edge_label_artists:
                self.edge_label_artists[edge].remove()
                del self.edge_label_artists[edge]
        # TODO remove edge data


    def _reverse_edges(self):
        edges = [self._reverse_edge_artists[artist] for artist in self._selected_artists if isinstance(artist, EdgeArtist)]
        edge_properties = [self._extract_edge_properties(self.edge_artists[edge]) for edge in edges]

        # delete old edges;
        # note this step has to be completed before creating new edges,
        # as bi-directional edges can pose a problem otherwise
        for edge in edges:
            self._delete_edge(edge)

        for edge, properties in zip(edges, edge_properties):
            self._add_edge(edge[1], edge[0], properties)

        self.edge_layout.get()
        self._update_edge_artists()


class EditableGraph(MutableGraph):
    """Extends :py:class:`InteractiveGraph` to support adding, deleting, and editing graph elements interactively.

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

        - :code:`networkx.Graph`, :code:`igraph.Graph`, or :code:`graph_tool.Graph` object

    *args, **kwargs
        Parameters passed through to :py:class:`InteractiveGraph`.
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
    :py:class:`InteractiveGraph`, :py:class:`EditableMultiGraph`

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not hasattr(self, 'node_label_artists'):
            self._initialize_empty_node_labels(kwargs)

        if not hasattr(self, 'edge_label_artists'):
            self._initialize_empty_edge_labels(kwargs)

        self._currently_writing_labels = False
        self._currently_writing_annotations = False

        # restore deprecated self.fig.canvas.manager.key_press method
        # https://github.com/matplotlib/matplotlib/issues/26713#issuecomment-1709938648
        from matplotlib import backend_bases
        self.fig.canvas.manager.key_press = backend_bases.key_press_handler


    def _initialize_empty_node_labels(self, kwargs):
        node_labels = {node : '' for node in self.nodes}
        self.node_label_fontdict = self._initialize_node_label_fontdict(
            kwargs.get('node_label_fontdict'), node_labels, kwargs.get('node_label_offset', (0., 0.)))
        self.node_label_offset, self._recompute_node_label_offsets =\
            self._initialize_node_label_offset(node_labels, kwargs.get('node_label_offset', (0., 0.)))
        if self._recompute_node_label_offsets:
            self._update_node_label_offsets()
        self.node_label_artists = dict()
        self.draw_node_labels(node_labels, self.node_label_fontdict)


    def _initialize_empty_edge_labels(self, kwargs):
        edge_labels = {edge : '' for edge in self.edges}
        self.edge_label_fontdict = self._initialize_edge_label_fontdict(kwargs.get('edge_label_fontdict'))
        self.edge_label_position = kwargs.get('edge_label_position', 0.5)
        self.edge_label_rotate = kwargs.get('edge_label_rotate', True)
        self.edge_label_artists = dict()
        self.draw_edge_labels(edge_labels, self.edge_label_position,
                              self.edge_label_rotate, self.edge_label_fontdict)


    def _on_key_press(self, event):
        if event.key == 'enter':
            if self._currently_writing_labels or self._currently_writing_annotations:
                self._terminate_writing()
            else:
                self._initiate_writing_labels()
        elif event.key == 'alt+enter':
            if self._currently_writing_annotations or self._currently_writing_labels:
                self._terminate_writing()
            else:
                self._initiate_writing_annotations()
        else:
            if self._currently_writing_labels:
                self._edit_labels(event.key)
            elif self._currently_writing_annotations:
                self._edit_annotations(event.key)
            else:
                super()._on_key_press(event)


    def _terminate_writing(self):
        self._currently_writing_labels = False
        self._currently_writing_annotations = False
        self.fig.canvas.manager.key_press_handler_id \
            = self.fig.canvas.mpl_connect('key_press_event', self.fig.canvas.manager.key_press)
        print('Finished writing.')


    def _initiate_writing_labels(self):
        self._currently_writing_labels = True
        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
        print('Initiated writing label(s).')


    def _initiate_writing_annotations(self):
        self._currently_writing_annotations = True
        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
        print('Initiated writing annotations(s).')


    def _edit_labels(self, key):
        for artist in self._selected_artists:
            if isinstance(artist, NodeArtist):
                self._edit_node_label(artist, key)
            elif isinstance(artist, EdgeArtist):
                self._edit_edge_label(artist, key)


    def _edit_node_label(self, artist, key):
        node = self.artist_to_key[artist]
        if node not in self.node_label_artists:
            # re-use a random offset to position node label;
            # we will improve the placement by updating all node label offsets
            self.node_label_offset[node] = next(iter(self.node_label_offset.values()))
            self._update_node_label_offsets()
            self.draw_node_labels({node : ''}, self.node_label_fontdict)

        self._edit_text_object(self.node_label_artists[node], key)


    def _edit_edge_label(self, artist, key):
        edge = self.artist_to_key[artist]
        if edge not in self.edge_label_artists:
            self.draw_edge_labels({edge : ''}, self.edge_label_position,
                                  self.edge_label_rotate, self.edge_label_fontdict)

        self._edit_text_object(self.edge_label_artists[edge], key)


    def _edit_annotations(self, key):
        for artist in self._selected_artists:
            if artist not in self.annotated_artists:
                if artist not in self.artist_to_annotation:
                    self.artist_to_annotation[artist] = ''
                self.annotated_artists.add(artist)
                placement = self._get_annotation_placement(artist)
                self._add_annotation(artist, *placement)

            self._edit_text_object(self.artist_to_text_object[artist], key)
            self.artist_to_annotation[artist] = self.artist_to_text_object[artist].get_text()


    def _edit_text_object(self, text_object, key):
        if len(key) == 1:
            text_object.set_text(text_object.get_text() + key)
        elif key == 'backspace':
            text_object.set_text(text_object.get_text()[:-1])
        self.fig.canvas.draw_idle()


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
        from matplotlib import backend_bases
        self.fig.canvas.manager.key_press = backend_bases.key_press_handler


    def _on_key_press(self, event):
        MutableMultiGraph._on_key_press(self, event)
        EditableGraph._on_key_press(self, event)
