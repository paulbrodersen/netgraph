#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
InteractiveGraph variants.
"""

import itertools
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backend_bases import key_press_handler

try:
    from ._main import InteractiveGraph, BASE_SCALE, DraggableGraph
    from ._line_supercover import line_supercover
    from ._artists import NodeArtist, EdgeArtist
except ValueError:
    from _main import InteractiveGraph, BASE_SCALE
    from _line_supercover import line_supercover


class InteractiveGrid(InteractiveGraph):
    """
    As InteractiveGraph, but node positions are fixed to a grid with unit spacing.

    Pressing 'g' will show the grid lines.
    Pressing 't' will show all tiles occupied by a node or crossed over by an egde.

    NOTE:
    -----
    For this class, the default netgraph node size and edge width are probably far too small for a medium sized graph.
    In my experience, for a graph with 20-50 nodes, a node size of 45 and an edge width of 15 tend to work well.
    Change your code accordingly. For example:

    g = InteractiveGrid(graph, node_size=45, edge_width=15)

    """

    def __init__(self, *args, **kwargs):

        super(InteractiveGrid, self).__init__(*args, **kwargs)

        self.show_tiles = False
        self.tiles = []
        self.gridlines = []
        self.show_grid = False

        self.fig.canvas.mpl_connect('key_press_event', self._on_key_toggle)


    def _get_node_positions(self, edge_list, **kwargs):
        """
        Initialise node positions to be on a grid with unit spacing.
        Prevent two points from occupying the same position.
        """

        node_positions = super(InteractiveGrid, self)._get_node_positions(edge_list, **kwargs)

        if len(kwargs) == 0: # i.e. using defaults

            # check if any two points will occupy the same grid point
            unique_grid_positions = set([(int(x), int(y)) for (x, y) in node_positions.values()])

            if len(unique_grid_positions) < len(node_positions):
                # rescale node positions such that each node occupies it's own grid point;

                keys   = np.array(list(node_positions.keys()))
                values = np.array(list(node_positions.values()))

                distances = np.sqrt(np.sum(np.power(values[None,:,:] - values[:,None,:], 2), axis=-1))
                distances = np.triu(distances, 1)
                sources, targets = np.where(distances)
                distances = distances[sources, targets]

                order = np.argsort(distances)
                for ii in order:
                    s = keys[sources[ii]]
                    t = keys[targets[ii]]
                    ps = node_positions[s]
                    pt = node_positions[t]
                    if np.all(np.isclose(ps.astype(np.int), pt.astype(np.int))):
                        minium_difference = np.min(np.abs(ps-pt))
                        scale = 1./minium_difference
                        break

                node_positions = {k : (v * scale).astype(np.int) for k,v in node_positions.items()}

        return node_positions


    def _on_release(self, event):

        if self._currently_dragging:

            nodes = [self._draggable_artist_to_node[artist] for artist in self._selected_artists]

            # set node positions to nearest grid point
            for node in nodes:
                x, y = self.node_positions[node]
                x = np.int(np.round(x))
                y = np.int(np.round(y))
                self.node_positions[node] = (x, y)

            self._update_nodes(nodes)
            self._update_edges(nodes)

            if self.show_grid:
                self._draw_grid()
            if self.show_tiles:
                self._draw_tiles(color='b', alpha=0.1)

            self.fig.canvas.draw_idle()

        super(InteractiveGrid, self)._on_release(event)


    def _draw_grid(self):
        xlim = [np.int(x) for x in self.ax.get_xlim()]
        for x in range(*xlim):
            line = self.ax.axvline(x, color='k', alpha=0.1, linestyle='--')
            self.gridlines.append(line)

        ylim = [np.int(y) for y in self.ax.get_ylim()]
        for y in range(*ylim):
            line = self.ax.axhline(y, color='k', alpha=0.1, linestyle='--')
            self.gridlines.append(line)


    def _remove_grid(self):
        for line in self.gridlines:
            line.remove()
        self.gridlines = []


    def _get_tile_positions(self):
        # find tiles through which a each edge crosses using the line supercover
        # (an extension of Bresenheims algorithm)
        tile_positions = []
        for (v0, v1) in self.edge_list:
            x0, y0 = self.node_positions[v0]
            x1, y1 = self.node_positions[v1]

            x0 = np.int(np.round(x0))
            y0 = np.int(np.round(y0))
            x1 = np.int(np.round(x1))
            y1 = np.int(np.round(y1))

            x, y = line_supercover(x0, y0, x1, y1)
            tile_positions.extend(zip(x.tolist(), y.tolist()))

        # remove duplicates
        tile_positions = list(set(tile_positions))
        return tile_positions


    def _draw_tiles(self, *args, **kwargs):
        # remove old tiles:
        # TODO: only remove tiles that are no longer in the set of positions
        self._remove_tiles()

        dx = 1. # TODO: generalise to arbitrary tile sizes
        dy = 1.
        positions = self._get_tile_positions()
        for (x, y) in positions:
            x -= dx/2.
            y -= dy/2.
            rect = Rectangle((x,y), dx, dy, *args, **kwargs)
            self.tiles.append(rect)
            self.ax.add_artist(rect)


    def _remove_tiles(self):
        for tile in self.tiles:
            tile.remove()
        self.tiles = []


    def _on_key_toggle(self, event):
        # print('you pressed', event.key, event.xdata, event.ydata)
        if event.key is 't':
            if self.show_tiles is False:
                self.show_tiles = True
                self._draw_tiles(color='b', alpha=0.1)
            else:
                self.show_tiles = False
                self._remove_tiles()
        if event.key is 'g':
            if self.show_grid is False:
                self.show_grid = True
                self._draw_grid()
            else:
                self.show_grid = False
                self._remove_grid()
        self.fig.canvas.draw_idle()


def demo_InteractiveGrid():
    n = 4
    adj = np.ones((n,n))
    adj = np.triu(adj, 1)
    pos = np.random.rand(n,2)
    pos[:,0] *= 10
    pos[:,1] *= 5
    pos = {ii:xy for ii, xy in enumerate(pos)}

    fig, ax = plt.subplots()
    ax.set(xlim=[0, 10], ylim=[0, 5])
    g = InteractiveGrid(adj, pos, ax=ax, node_size=15.)

    return g


class InteractiveHypergraph(InteractiveGraph):
    """
    As InteractiveGraph, but nodes can be combined into a hypernode.

    Pressing 'c' will fuse selected node artists into a single node.
    """

    def __init__(self, *args, **kwargs):

        super(InteractiveHypergraph, self).__init__(*args, **kwargs)

        # bookkeeping
        self.hypernode_to_nodes = dict()

        # for redrawing after fusion
        self.kwargs = kwargs

        # set up ability to trigger fusion by key-press
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_group_ungroup)


    def _on_key_group_ungroup(self, event):

        if event.key == 'c':
            if len(self._selected_artists) > 1:
                nodes = [self._draggable_artist_to_node[artist] for artist in self._selected_artists]
                self._deselect_all_artists()
                self._combine(nodes)
            else:
                print("Only a single artist selected! Nothing to combine.")


    def _combine(self, nodes):

        # create hypernode ID
        # hypernode = _find_unused_int(self.edge_list)
        hypernode = tuple(set(nodes))

        # bookkeeping
        self.hypernode_to_nodes[hypernode] = nodes

        # create hypernode
        self._create_hypernode(nodes, hypernode)

        # create corresponding edges
        new_edge_list = self._transfer_edges_to_hypernode(self.edges, nodes, hypernode)
        new_edges = [edge for edge in new_edge_list if not edge in self.edges]
        old_edges = [edge for edge in self.edges if not edge in new_edge_list]
        self._create_hypernode_edges(old_edges, new_edges)

        # update graph structure
        self.edges = list(set(new_edge_list))

        # clean up data structures and remove obsolote artists
        for edge in old_edges:
            self._delete_edge(edge)

        for node in nodes:
            self._delete_node(node)

        # draw new state
        self.fig.canvas.draw_idle()


    def _create_hypernode(self, nodes, hypernode, combine_properties=partial(np.mean, axis=0)):
        """
        Combine properties of nodes that will form hypernode.
        Draw hypernode.
        """

        # combine node / node artist properties
        pos             = combine_properties([self.node_positions[node]                    for node in nodes])
        node_size       = combine_properties([self.node_edge_artists[node].radius          for node in nodes])
        node_edge_width = combine_properties([self.node_face_artists[node].radius          for node in nodes]); node_edge_width = node_size - node_edge_width
        node_color      = combine_properties([self.node_face_artists[node].get_facecolor() for node in nodes]) # NB: this only makes sense for a gray cmap
        node_edge_color = combine_properties([self.node_edge_artists[node].get_facecolor() for node in nodes]) # NB: this only makes sense for a gray cmap
        node_alpha      = combine_properties([self.node_face_artists[node].get_alpha()     for node in nodes])
        node_edge_alpha = combine_properties([self.node_edge_artists[node].get_alpha()     for node in nodes])

        # update data
        self.node_positions[hypernode] = pos
        self._base_alpha[hypernode] = node_alpha

        # draw hypernode
        self.draw_nodes({hypernode:pos}, # has to be {} not dict()!
                        node_size=node_size / BASE_SCALE,
                        node_edge_width=node_edge_width / BASE_SCALE,
                        node_color=node_color,
                        node_edge_color=node_edge_color,
                        node_alpha=node_alpha,
                        node_edge_alpha=node_edge_alpha,
                        ax=self.ax)

        # add to draggable artists
        hypernode_artist = self.node_face_artists[hypernode]
        self._draggable_artists.append(hypernode_artist)
        self._node_to_draggable_artist[hypernode] = hypernode_artist
        self._draggable_artist_to_node[hypernode_artist] = hypernode
        self._base_alpha[hypernode_artist] = hypernode_artist.get_alpha()

        if hasattr(self, 'node_labels'):

            # # TODO: call to `input` results in unresponsive plot and terminal; fix / find workaround
            # hypernode_label = input("Please provide a new label for the hypernode and press enter (default {}):\n".format(hypernode))
            # if hypernode_label == '':
            #     hypernode_label = str(hypernode)
            hypernode_label = [self.node_label[node] for node in nodes]
            hypernode_label = ',\n'.join(hypernode_label)

            self.node_labels[hypernode] = hypernode_label

            if hasattr(self, 'node_label_font_size'):
                self.draw_node_labels({hypernode:hypernode_label}, {hypernode:pos}, node_label_font_size=self.node_label_font_size) # has to be {} not dict()!
            else:
                self.draw_node_labels({hypernode:hypernode_label}, {hypernode:pos})                                                 # has to be {} not dict()!


    # def _delete_node(self, node):
    #     del self.node_positions[node]
    #     self.node_face_artists[node].remove()
    #     del self.node_face_artists[node]
    #     self.node_edge_artists[node].remove()
    #     del self.node_edge_artists[node]

    #     if hasattr(self, 'node_labels'):
    #         self.node_label_artists[node].remove()
    #         del self.node_label_artists[node]
    #         del self.node_labels[node]
    def _delete_node(self, node):
        self.node_face_artists[node].set_visible(False)
        self.node_edge_artists[node].set_visible(False)

        if hasattr(self, 'node_labels'):
            self.node_label_artists[node].set_visible(False)

        artist = self._node_to_draggable_artist[node]
        del self._draggable_artist_to_node[artist]
        del self._node_to_draggable_artist[node]


    def _transfer_edges_to_hypernode(self, edge_list, nodes, hypernode):
        """
        Note:
        - does not remove self-loops
        - may contain duplicate edges after fusion
        """

        # replace nodes in `nodes` with hypernode
        new_edge_list = []
        for (source, target) in edge_list:
            if source in nodes:
                source = hypernode
            if target in nodes:
                target = hypernode
            new_edge_list.append((source, target))

        return new_edge_list


    def _create_hypernode_edges(self, old_edges, new_edges, combine_properties=partial(np.mean, axis=0)):
        """
        For each unique new edge, take corresponding old edges.
        Create new edge artists based on properties of corresponding old edge artists.
        """

        # find edges that are being combined
        new_to_old = dict()
        for new_edge, old_edge in zip(new_edges, old_edges):
            try:
                new_to_old[new_edge].append(old_edge)
            except KeyError:
                new_to_old[new_edge] = [old_edge]

        # combine edge properties
        edge_width = dict()
        edge_color = dict()
        edge_alpha = dict()
        for new_edge, old_edges in new_to_old.items():
            # filter old_edges: self-loops have no edge artists
            old_edges = [(source, target) for (source, target) in old_edges if source != target]
            # combine properties
            edge_width[new_edge] = combine_properties([self.edge_artists[edge].width           for edge in old_edges])  / BASE_SCALE
            edge_color[new_edge] = combine_properties([self.edge_artists[edge].get_facecolor() for edge in old_edges]) # NB: this only makes sense for a gray cmap; combine weights instead?
            # edge_alpha[new_edge] = combine_properties([self.edge_artists[edge].get_alpha()     for edge in old_edges]) # TODO: .get_alpha() returns None?

        # zorder = _get_zorder(self.edge_color) # TODO: fix, i.e. get all edge colors, determine order

        # remove duplicates in new_edges
        new_edges = new_to_old.keys()

        # don't plot self-loops
        new_edges = [(source, target) for (source, target) in new_edges if source != target]

        self.draw_edges(new_edges,
                        node_positions=self.node_positions,
                        edge_width=edge_width,
                        edge_color=edge_color,
                        # edge_alpha=edge_alpha,
                        ax=self.ax)


    def _delete_edge(self, edge):

        # del self.edge_weight[edge]            # TODO: get / set property for hyperedges; otherwise these raises KeyError
        # del self.edge_color[edge]             # TODO: get / set property for hyperedges; otherwise these raises KeyError
        # del self.edge_zorder[edge]            # TODO: get / set property for hyperedges; otherwise these raises KeyError

        source, target = edge
        if source != target: # i.e. skip self-loops as they have no corresponding artist
            self.edge_artists[edge].remove()
            del self.edge_artists[edge]

            # if hasattr(self, 'edge_labels'):
            #     del self.edge_labels[edge]        # TODO: get / set property for hyperedges; otherwise these raises KeyError
            #     edge_label_artists[edge].remove() # TODO: get / set property for hyperedges; otherwise these raises KeyError
            #     del self.edge_label_artists[edge] # TODO: get / set property for hyperedges; otherwise these raises KeyError


class NascentEdge(plt.Line2D):
    def __init__(self, p1, p2):
        super().__init__(p1, p2, color='lightgray', linestyle='--')


class MutableGraph(InteractiveGraph):
    """
    Add or remove nodes and edges:
    -------------------------------
    - Double clicking on two nodes successively will create an edge between them.
    - Pressing 'insert' or '+' will add a new node to the graph.
    - Pressing 'delete' or '-' will remove selected nodes and edges.
    - Pressing '@' will reverse the direction of selected edges.

    When adding a new node, the properties of the last selected node will be used to style the node artist.
    Ditto for edges. If no node or edge has been previously selected the first created node or edge artist will be used.

    See also:
    ---------
    InteractiveGraph
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self._reverse_node_artists = {artist : node for node, artist in self.node_artists.items()}
        self._reverse_edge_artists = {artist : edge for edge, artist in self.edge_artists.items()}
        self._last_selected_node_properties = self._extract_node_properties(next(iter(self.node_artists.values())))
        self._last_selected_edge_properties = self._extract_edge_properties(next(iter(self.edge_artists.values())))
        self._nascent_edge_source = None
        self._nascent_edge = None

        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)


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

        for artist in self._clickable_artists:
            if artist.contains(event)[0]:
                self._extract_artist_properties(artist)
                break

        if event.dblclick:
            for node, artist in self.node_artists.items():
                if artist.contains(event)[0]:
                    if self._nascent_edge_source is not None: # NB: node can have ID 0!
                        # connect edge to target node
                        if (self._nascent_edge_source, node) not in self.edges:
                            self._add_edge((self._nascent_edge_source, node))
                        else:
                            print("Edge already exists!")
                        self._nascent_edge_source = None
                        self._nascent_edge.remove()
                        self._nascent_edge = None
                    else:
                        # initiate edge
                        self._nascent_edge_source = node
                        x0, y0 = self.node_positions[node]
                        self._nascent_edge = NascentEdge((x0, x0), (y0, y0))
                        self.ax.add_artist(self._nascent_edge)
                    break
            else:
                if self._nascent_edge:
                    self._nascent_edge_source = None
                    self._nascent_edge.remove()
                    self._nascent_edge = None


    def _extract_artist_properties(self, artist):
        if isinstance(artist, NodeArtist):
            self._last_selected_node_properties = self._extract_node_properties(artist)
        elif isinstance(artist, EdgeArtist):
            self._last_selected_edge_properties = self._extract_edge_properties(artist)


    def _extract_node_properties(self, node_artist):
        return dict(
            shape     = node_artist.shape,
            radius    = node_artist.radius,
            facecolor = node_artist.get_facecolor(),
            edgecolor = self._base_edgecolor[node_artist],
            linewidth = self._base_linewidth[node_artist],
            alpha     = self._base_alpha[node_artist],
            zorder    = node_artist.get_zorder()
        )


    def _extract_edge_properties(self, edge_artist):
        return dict(
            width       = edge_artist.width,
            facecolor   = edge_artist.get_facecolor(),
            alpha       = self._base_alpha[edge_artist],
            head_length = edge_artist.head_length,
            head_width  = edge_artist.head_width,
            edgecolor   = self._base_edgecolor[edge_artist],
            linewidth   = self._base_linewidth[edge_artist],
            offset      = edge_artist.offset, # TODO: need to get node_size of target node instead
            curved      = edge_artist.curved,
            zorder      = edge_artist.get_zorder(),
        )


    def _on_motion(self, event):
        super()._on_motion(event)

        if self._nascent_edge:
            x, y = self._nascent_edge.get_data()
            self._nascent_edge.set_data(((x[0], event.xdata), (y[0], event.ydata)))
            self.fig.canvas.draw_idle()


    def _add_node(self, event):
        if event.inaxes != self.ax:
            print('Position outside of axis limits! Cannot create node.')
            return

        # create node ID; use smallest unused int
        node = 0
        while node in self.node_positions.keys():
            node += 1

        # get position of cursor place node at cursor position
        pos = event.xdata, event.ydata

        # copy attributes of last selected artist;
        # if none is selected, use a random artist
        if self._selected_artists:
            node_properties = self._extract_node_properties(self._selected_artists[-1])
        else:
            node_properties = self._last_selected_node_properties

        artist = NodeArtist(xy = pos, **node_properties)

        self._reverse_node_artists[artist] = node

        # Update data structures in parent classes:
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
        self.ax.add_patch(artist)
        # self.node_label_artists # TODO (potentially)
        # self.node_label_offset  # TODO (potentially)


    def _delete_nodes(self):
        # translate selected artists into nodes
        nodes = [self._reverse_node_artists[artist] for artist in self._selected_artists if isinstance(artist, NodeArtist)]

        # delete edges to and from selected nodes
        edges = [(source, target) for (source, target) in self.edges if ((source in nodes) or (target in nodes))]
        for edge in edges:
            self._delete_edge(edge)

        # delete nodes
        for node in nodes:
            self._delete_node(node)


    def _delete_node(self, node):
        # print(f"Deleting node {node}.")
        artist = self.node_artists[node]

        del self._reverse_node_artists[artist]

        # Update data structures in parent classes:
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
        self._selected_artists.remove(artist)
        del self._base_linewidth[artist]
        del self._base_edgecolor[artist]
        # 3c) EmphasizeOnHover
        self.emphasizeable_artists.remove(artist)
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
        artist.remove()


    def _add_edge(self, edge, edge_properties=None):
        source, target = edge
        path = np.array([self.node_positions[source], self.node_positions[target]])

        # create artist
        if not edge_properties:
            edge_properties = self._last_selected_edge_properties

        if (target, source) in self.edges:
            shape = 'right'
            self.edge_artists[(target, source)].shape = 'right'
            self.edge_artists[(target, source)]._update_path()
        else:
            shape = 'full'

        artist = EdgeArtist(midline=path, shape=shape, **edge_properties)

        self._reverse_edge_artists[artist] = edge

        # update data structures in parent classes
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
        self.edge_paths[edge] = path
        self.edge_artists[edge] = artist
        self.ax.add_patch(artist)


    def _delete_edges(self):
        edges = [self._reverse_edge_artists[artist] for artist in self._selected_artists if isinstance(artist, EdgeArtist)]
        for edge in edges:
            self._delete_edge(edge)


    def _delete_edge(self, edge):
        artist = self.edge_artists[edge]
        del self._reverse_edge_artists[artist]

        source, target = edge
        if (target, source) in self.edges:
            self.edge_artists[(target, source)].shape = 'full'
            self.edge_artists[(target, source)]._update_path()

        # update data structures in parent classes
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
        del self._base_alpha[artist]
        # 3d) AnnotateOnClick
        if artist in self.artist_to_annotation:
            del self.artist_to_annotation[artist]
        # 4) BaseGraph
        self.edges.remove(edge)
        del self.edge_paths[edge]
        del self.edge_artists[edge]
        if hasattr(self, 'edge_label_artists'):
            if edge in self.edge_label_artists:
                self.edge_label_artists[edge].remove()
                del self.edge_label_artists[edge]
        # TODO remove edge data
        artist.remove()


    def _reverse_edges(self):
        edges = [self._reverse_edge_artists[artist] for artist in self._selected_artists if isinstance(artist, EdgeArtist)]
        edge_properties = [self._extract_edge_properties(self.edge_artists[edge]) for edge in edges]

        # delete old edges;
        # note this step has to be completed before creating new edges,
        # as bi-directional edges can pose a problem otherwise
        for edge in edges:
            self._delete_edge(edge)

        for edge, properties in zip(edges, edge_properties):
            self._add_edge(edge[::-1], properties)


class EditableGraph(MutableGraph):
    """
    Add or remove nodes and edges:
    -------------------------------
    - Double clicking on two nodes successively will create an edge between them.
    - Pressing 'insert' or '+' will add a new node to the graph.
    - Pressing 'delete' or '-' will remove selected nodes and edges.
    - Pressing '&' will reverse the direction of selected edges.

    When adding a new node, the properties of the last selected node will be used to style the node artist.
    Ditto for edges. If no node or edge has been previously selected, the first created node or edge artist will be used.

    Create or edit labels and annotations:
    --------------------------------------
    - To create or edit a node or edge label, select the node (or edge) artist, press the 'enter' key, and type.
    - To create or edit an annotation, select the node (or edge) artist, press 'alt'+'enter', and type.

    Terminate either action by pressing 'enter' or 'alt'+'enter' a second time.

    See also:
    ---------
    InteractiveGraph
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
