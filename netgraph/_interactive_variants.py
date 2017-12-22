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

try:
    from ._main import InteractiveGraph, BASE_EDGE_WIDTH, BASE_NODE_SIZE, test
    from ._line_supercover import line_supercover
except ValueError:
    from _main import InteractiveGraph, BASE_EDGE_WIDTH, BASE_NODE_SIZE, test
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

                keys = node_positions.keys()
                values = np.array(node_positions.values())

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
        new_edge_list = self._transfer_edges_to_hypernode(self.edge_list, nodes, hypernode)
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
                        node_size=node_size / BASE_NODE_SIZE,
                        node_edge_width=node_edge_width / BASE_NODE_SIZE,
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
            edge_width[new_edge] = combine_properties([self.edge_artists[edge].width           for edge in old_edges])  / BASE_EDGE_WIDTH
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



if __name__ == '__main__':

    # g = test(InteractiveClass=InteractiveGrid, node_size=45, edge_width=15)
    g = test(InteractiveClass=InteractiveHypergraph)
    plt.show()
