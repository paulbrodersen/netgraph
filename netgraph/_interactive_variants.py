#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
InteractiveGraph variants.
"""

import numpy as np
import matplotlib.pyplot as plt
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


if __name__ == '__main__':

    from _main import test
    g = test(InteractiveClass=InteractiveGrid)
    plt.show()
