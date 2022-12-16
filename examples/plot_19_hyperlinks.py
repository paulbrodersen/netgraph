#!/usr/bin/env python
"""Hyperlinks
==========

Non-interactive figures
-----------------------

Netgraph uses matplotlib to draw all figure elements, and `matplotib
has only limited support for hyperlinks`_. Specifically, you have to use the
SVG backend or export the figure to SVG.

.. _matplotib has only limited support for hyperlinks: https://matplotlib.org/stable/gallery/misc/hyperlinks_sgskip.html

"""

import matplotlib.pyplot as plt

from netgraph import Graph

fig, ax = plt.subplots()
g = Graph([(0, 1)], ax=ax)

g.node_artists[0].set_url("https://www.google.com")
g.edge_artists[(0, 1)].set_url("https://www.stackoverflow.com")

fig.savefig('image.svg')

################################################################################
# Opening the image.svg file in a browser of your choice, you can click on
# the plot elements to open the corresponding web pages.
#
# Interactive figures
# -------------------
#
# Technically, matplotlib does not support hyperlinks when using an
# interactive backend. However, hyperlink behaviour can easily be
# emulated using `matplotlib pick-events`_ and the python in-built
# :code:`webbrowser` module. Note that these hyperlinks are not
# available when exporting the figure (unless the figure is exported
# to SVG). Clicking on the PNG image embedded below will hence have
# no effect.
#
# .. _matplotlib pick-events: https://matplotlib.org/stable/gallery/event_handling/pick_event_demo.html

import webbrowser
import matplotlib.pyplot as plt

from netgraph import Graph

def on_pick(event):
    webbrowser.open(event.artist.get_url())

fig, ax = plt.subplots()

fig.canvas.mpl_connect('pick_event', on_pick)

g = Graph([(0, 1)],
          node_labels={1 : "https://www.github.com"},
          node_label_offset=0.1,
          edge_labels={(0, 1) : "https://matplotlib.org/"},
          ax=ax)

g.node_artists[0].set_picker(True)
g.node_artists[0].set_url("https://www.google.com")

g.edge_artists[(0, 1)].set_picker(10) # increases the pick radius
g.edge_artists[(0, 1)].set_url("https://www.stackoverflow.com")

g.node_label_artists[1].set_picker(True)
g.node_label_artists[1].set_url(g.node_label_artists[1].get_text()) # assumes the label text defines the link target

g.edge_label_artists[(0, 1)].set_picker(True)
g.edge_label_artists[(0, 1)].set_url(g.edge_label_artists[(0, 1)].get_text()) # ditto

plt.show()
