#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Examples for netgraph
"""

import numpy as np
import matplotlib.pyplot as plt; plt.ion()

try:
    reload         # Python 2
except NameError:  # Python 3
    from importlib import reload 

import netgraph; reload(netgraph)

try:
    raw_input      # Python 2
except NameError:  # Python 3
    raw_input = input

FDIR = "./figures/"

def plot(title, **kwargs):
    fig, ax = plt.subplots(1,1)
    graph = netgraph.test(ax=ax, **kwargs)
    fig.tight_layout()
    fig.canvas.draw()
    raw_input(title)
    graph._update_view()
    fig.savefig(FDIR + title.replace(' ', '_') + '.pdf')
    fig.savefig(FDIR + title.replace(' ', '_') + '.svg')
    plt.close()

if __name__ == "__main__":

    plot('Directed', directed=True, interactive=True)
    plot('Undirected', directed=False, interactive=True)

    plot('Weighted', weighted=True, interactive=True)
    plot('Unweighted', weighted=False, interactive=True)

    plot('Show node labels', show_node_labels=True, interactive=True)
    plot('Show edge labels', show_edge_labels=True, interactive=True)

    plot('Positive edge weights only', strictly_positive=True, interactive=True)
    plot('Positive and negative edge weights', strictly_positive=False, interactive=True)
