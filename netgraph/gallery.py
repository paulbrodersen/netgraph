#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Examples for netgraph
"""

import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import netgraph

FDIR = "./figures/"

def plot(title, **kwargs):
    fig, ax = plt.subplots(1,1)
    graph = netgraph.test(ax=ax, **kwargs)
    fig.tight_layout()
    fig.canvas.draw()
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
