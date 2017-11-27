#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
InteractiveGraph variants.
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools

from functools import partial
from matplotlib.patches import Rectangle

from _main import InteractiveGraph, BASE_EDGE_WIDTH, BASE_NODE_SIZE
from _line_supercover import line_supercover


class InteractiveGrid(InteractiveGraph):

    def __init__(self, *args, **kwargs):

        super(InteractiveGrid, self).__init__(*args, **kwargs)

        self.show_tiles = False
        self.tiles = []
        self.gridlines = []
        self.show_grid = False

        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _on_release(self, event):

        if self._currently_dragging:
            for key in self._selected_artists.keys():
                x, y = self.node_positions[key]
                x = np.int(np.round(x))
                y = np.int(np.round(y))
                self._move_node(key, (x,y))

            self._update_edges()

            if self.show_grid:
                self._draw_grid()

            if self.show_tiles:
                self._draw_tiles(color='b', alpha=0.1)

        super(InteractiveGrid, self)._on_release(event)

        self.fig.canvas.draw_idle()

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

    def _on_key(self, event):
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

    def __init__(self, *args, **kwargs):

        super(InteractiveHypergraph, self).__init__(*args, **kwargs)

        # bookkeeping
        self.hypernode_to_nodes = dict()

        # for redrawing after fusion
        self.kwargs = kwargs

        # set up ability to trigger fusion by key-press
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)


    def _on_key(self, event):

        if hasattr(super(InteractiveHypergraph, self), '_on_key'):
            super(InteractiveHypergraph, self)._on_key(event)

        if event.key == 'c':
            nodes = self._selected_artists.keys()
            self._deselect_artists()
            self._fuse(nodes)


    def _fuse(self, nodes):

        # create hypernode ID
        hypernode = _find_unused_int(self.edge_list)

        # create hypernode
        self._create_hypernode(nodes, hypernode)

        # create corresponding edges
        new_edge_list = _fuse_nodes_into_hypernode(self.edge_list, nodes, hypernode)
        new_edges = [edge for edge in new_edge_list if not edge in self.edge_list]
        old_edges = [edge for edge in self.edge_list if not edge in new_edge_list]
        self._create_hypernode_edges(old_edges, new_edges)

        # update graph structure
        self.edge_list = list(set(new_edge_list))

        # clean up data structures and remove obsolote artists
        for edge in old_edges:
            self._delete_edge(edge)

        for node in nodes:
            self._delete_node(node)

        # draw new state
        self.fig.canvas.draw_idle()


    def _create_hypernode(self, nodes, hypernode, fuse_properties=partial(np.mean, axis=0)):
        """
        Combine properties of nodes that will form hypernode.
        Draw hypernode.
        """

        # combine node / node artist properties
        pos             = fuse_properties([self.node_positions[node]                    for node in nodes])
        node_size       = fuse_properties([self.node_edge_artists[node].radius          for node in nodes])
        node_edge_width = fuse_properties([self.node_face_artists[node].radius          for node in nodes]); node_edge_width = node_size - node_edge_width
        node_color      = fuse_properties([self.node_face_artists[node].get_facecolor() for node in nodes]) # NB: this only makes sense for a gray cmap
        node_edge_color = fuse_properties([self.node_edge_artists[node].get_facecolor() for node in nodes]) # NB: this only makes sense for a gray cmap
        node_alpha      = fuse_properties([self.node_face_artists[node].get_alpha()     for node in nodes])
        node_edge_alpha = fuse_properties([self.node_edge_artists[node].get_alpha()     for node in nodes])

        # update data
        self.node_positions[hypernode] = pos
        self._alpha[hypernode] = node_alpha

        # draw hypernode
        self.draw_nodes({hypernode: pos},
                        node_size=node_size / BASE_NODE_SIZE,
                        node_edge_width=node_edge_width / BASE_NODE_SIZE,
                        node_color=node_color,
                        node_edge_color=node_edge_color,
                        node_alpha=node_alpha,
                        node_edge_alpha=node_edge_alpha,
                        ax=self.ax)

        if hasattr(self, 'node_labels'):
            self.node_labels[hypernode] = hypernode
            self.draw_node_labels(dict(hypernode=hypernode)) # TODO: pass in kwargs


    def _delete_node(self, node):
        del self.node_positions[node]
        self.node_face_artists[node].remove()
        del self.node_face_artists[node]
        self.node_edge_artists[node].remove()
        del self.node_edge_artists[node]

        if hasattr(self, 'node_labels'):
            self.node_label_artist[node].remove()
            del self.node_label_artist[node]
            del self.node_labels[node]


    def _create_hypernode_edges(self, old_edges, new_edges, fuse_properties=partial(np.mean, axis=0)):
        """
        For each unique new edge, take corresponding old edges.
        Create new edge artists based on properties of corresponding old edge artists.
        """

        # find edges that are being fused
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
            edge_width[new_edge] = fuse_properties([self.edge_artists[edge].width           for edge in old_edges])  / BASE_EDGE_WIDTH
            edge_color[new_edge] = fuse_properties([self.edge_artists[edge].get_facecolor() for edge in old_edges]) # NB: this only makes sense for a gray cmap; combine weights instead?
            # edge_alpha[new_edge] = fuse_properties([self.edge_artists[edge].get_alpha()     for edge in old_edges]) # TODO: .get_alpha() returns None?

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


def _find_unused_int(iterable):
    unique = np.unique(iterable)
    for ii in itertools.count():
        if not (ii in unique):
            break
    return ii


def _fuse_nodes_into_hypernode(edge_list, nodes, hypernode):
    """
    TODO: rename

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


if __name__ == '__main__':

    from _main import test
    g = test(InteractiveClass=InteractiveGrid)
    g = test(InteractiveClass=InteractiveHypergraph)
    plt.show()
