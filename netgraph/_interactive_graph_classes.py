#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implements the InteractiveGraph, MutableGraph, and EditableGraph classes,
as well as the following helper classes:

  - ClickableArtists
  - SelectableArtists
  - DraggableArtists
  - DraggableGraph
  - DraggableGraphWithGridMode
  - EmphasizeOnHover
  - EmphasizeOnHoverGraph
  - AnnotateOnClick
  - AnnotateOnClickGraph
  - TableOnClick
  - TableOnClickGraph

"""

import warnings
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backend_bases import key_press_handler

from ._graph_classes import BASE_SCALE, Graph
from ._artists import (
    NodeArtist,
    CircularNodeArtist,
    RegularPolygonNodeArtist,
    EdgeArtist,
)
from ._utils import (
    _get_angle,
    _get_interior_angle_between,
    _get_orthogonal_unit_vector,
    _get_point_along_spline,
    _get_tangent_at_point,
)
from ._parser import is_order_zero, is_empty, parse_graph


DEFAULT_EDGE = (0, 1)
DEFAULT_EDGE_KEY = 0


class ClickableArtists(object):
    """Implements selection of matplotlib artists via the mouse left click (+/- ctrl or command key).

    Notes
    -----
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

        if plt.get_backend() == 'MacOSX':
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
    """Augments :py:class:`ClickableArtists` with a rectangle selector.

    Notes
    -----
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
    """Augments :py:class:`SelectableArtists` to support dragging of artists by holding the left mouse button.

    Notes
    -----
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
    """
    Augments :py:class:`Graph` to support selection and dragging of node artists with the mouse.

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
    :py:class:`Graph`, :py:class:`DraggableArtists`, :py:class:`InteractiveGraph`

    """

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

        edge_paths = self.edge_layout.get(nodes, approximate=True)
        self._update_edge_artists(edge_paths)
        if hasattr(self, 'edge_label_artists'):
            self._update_edge_label_positions(edge_paths.keys())

        self.fig.canvas.draw_idle()


    def _get_stale_nodes(self):
        return [self._draggable_artist_to_node[artist] for artist in self._selected_artists if artist in self._draggable_artists]


    def _update_node_positions(self, nodes, cursor_position):
        for node in nodes:
            self.node_positions[node] = cursor_position + self._offset[self.node_artists[node]]


    def _on_release(self, event):
        if self._currently_dragging:
            edge_paths = self.edge_layout.get()
            self._update_edge_artists(edge_paths)
            if hasattr(self, 'edge_label_artists'): # move edge labels
                self._update_edge_label_positions(edge_paths.keys())

        super()._on_release(event)


    # def _on_resize(self, event):
    #     if hasattr(self, 'node_labels'):
    #         self.draw_node_labels(self.node_labels)
    #         # print("As node label font size was not explicitly set, automatically adjusted node label font size to {:.2f}.".format(self.node_label_font_size))


class DraggableGraphWithGridMode(DraggableGraph):
    """
    Adds a grid-mode to :py:class:`DraggableGraph`, in which node positions are fixed to a grid.
    To activate, press the letter 'g'.

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
    :py:class:`DraggableGraph`, :py:class:`InteractiveGraph`

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

            edge_paths = self.edge_layout.get()
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
            artist = self._is_on_artist(event)
            if artist:
                self._emphasize(artist)
            elif self.deemphasized_artists:
                self._reset_emphasis()


    def _is_on_artist(self, event):
        for artist in self.emphasizeable_artists:
            if artist.contains(event)[0]: # returns two arguments for some reason
                return artist
        return None


    def _emphasize(self, artist_to_emphasize):
        for artist in self.emphasizeable_artists:
            if artist is not artist_to_emphasize:
                artist.set_alpha(self._base_alpha[artist]/5)
                self.deemphasized_artists.append(artist)
        self.fig.canvas.draw_idle()


    def _reset_emphasis(self):
        for artist in self.deemphasized_artists:
            try:
                artist.set_alpha(self._base_alpha[artist])
            except KeyError:
                # This is a workaround for a bug in MutableGraph that occurs when:
                # 1) The user selects an artist.
                # 2) The user moves the mouse and hovers over a different artist that results in de-emphasizing the selected artist.
                # 3) The user deletes the selected artist.
                # In theory, self.deemphasized_artists should be updated on node/edge removal.
                # In practice, it still contains the artist when this function is called.
                pass
        self.deemphasized_artists.clear()
        self.fig.canvas.draw_idle()


class EmphasizeOnHoverGraph(Graph, EmphasizeOnHover):
    """Combines :py:class:`EmphasizeOnHover` with the :py:class:`Graph` class
    such that nodes are emphasized when hovering over them with the mouse.

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

    mouseover_highlight_mapping : dict or None, default None
        Determines which nodes and/or edges are highlighted when hovering over any given node or edge.
        The keys of the dictionary are node and/or edge IDs, while the values are iterables of node and/or edge IDs.
        If the parameter is None, a default dictionary is constructed, which maps

        - edges to themselves as well as their source and target nodes, and
        - nodes to themselves as well as their immediate neighbours and any edges between them.

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
    :py:class:`Graph`, :py:class:`EmphasizeOnHover`, :py:class:`InteractiveGraph`

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


    def _emphasize(self, selected_artist):
        key = self.artist_to_key[selected_artist]
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
        angle = _get_angle(dx, dy, radians=False) % 360

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
    """Combines :py:class:`AnnotateOnClick` with the :py:class:`Graph` class
    such that nodes or edges can have toggleable annotations.

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

    annotations : dict
        Mapping of nodes or edges to strings or dictionaries, the annotations.
        The visibility of the annotations can be toggled on or off by clicking on the corresponding node or edge.

        .. code-block::

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
    :py:class:`Graph`, :py:class:`AnnotateOnClick`, :py:class:`InteractiveGraph`

    """

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
        if _get_interior_angle_between(orthogonal_vector, vector_pointing_outwards, radians=False) > 90:
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
    """Combines :py:class:`TableOnClick` with :py:class:`Graph`
    such that nodes or edges can have toggleable tabular annotations.

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

    tables : dict node/edge : pandas dataframe
        Mapping of nodes and/or edges to pandas dataframes.
        The visibility of the tables that can toggled on or off by clicking on the corresponding node or edge.
    table_kwargs : dict
        Keyword arguments passed to matplotlib.pyplot.table.
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
    :py:class:`Graph`, :py:class:`TableOnClick`, :py:class:`InteractiveGraph`

    """

    def __init__(self, *args, **kwargs):
        Graph.__init__(self, *args, **kwargs)
        self._setup_table_annotations(*args, **kwargs)


    def _setup_table_annotations(self, *args, **kwargs):
        if 'tables' in kwargs:
            artist_to_table = self._map_artist_to_table(kwargs['tables'])
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
    """Extends the :py:class:`Graph` to support node placement with the mouse,
    emphasis of graph elements when hovering over them, and toggleable annotations.

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

        - :code:`networkx.Graph`, :code:`igraph.Graph`, or :code:`graph_tool.Graph` object

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
    :py:class:`Graph`, :py:class:`EditableGraph`, :py:class:`InteractiveMultiGraph`, :py:class:`InteractiveArcDiagram`

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
        if self.autoscale_node_labels:
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
        for artist in self._selected_artists[::-1]:
            if isinstance(artist, NodeArtist):
                node_properties = self._extract_node_properties(artist)
                node_type = type(artist)
                break
        else:
            node_properties = self._last_selected_node_properties
            node_type = self._last_selected_node_type

        artist = node_type(xy=pos, **node_properties)
        self.ax.add_patch(artist)
        self._expand_node_data_structures(node, artist, pos)

        return node


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

        return edge


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
        if hasattr(self, 'edge_label_artists'):
            self.draw_edge_labels(
                {edge : ""},
                self.edge_label_position,
                self.edge_label_rotate,
                self.edge_label_fontdict,
            )


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
        self.fig.canvas.manager.key_press = key_press_handler


    def _initialize_empty_node_labels(self, kwargs):
        node_labels = {node : '' for node in self.nodes}
        self.node_label_fontdict = self._initialize_node_label_fontdict(
            kwargs.get('node_label_fontdict', None))
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
