.. Netgraph documentation master file, created by
   sphinx-quickstart on Mon Feb 21 12:02:02 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Netgraph
========

Netgraph is a python library for creating publication quality plots of networks.


Installation
------------

.. code-block:: shell

    pip install netgraph


Quickstart
----------

.. code-block:: python

    import matplotlib.pyplot as plt
    from netgraph import Graph, InteractiveGraph, EditableGraph

    # Several graph formats are supported:

    # 1) edge lists
    graph_data = [(0, 1), (1, 2), (2, 0)]

    # 2) edge list with weights
    graph_data = [(0, 1, 0.2), (1, 2, -0.4), (2, 0, 0.7)]

    # 3) full rank matrices
    import numpy
    graph_data = np.random.rand(10, 10)

    # 4) networkx Graph and DiGraph objects (MultiGraph objects are not supported, yet)
    import networkx
    graph_data = networkx.karate_club_graph()

    # 5) igraph.Graph objects
    import igraph
    graph_data = igraph.Graph.Famous('Zachary')

    # Create a non-interactive plot:
    Graph(graph_data)
    plt.show()

    # Create an interactive plot, in which the nodes can be re-positioned with the mouse.
    # NOTE: you must retain a reference to the plot instance!
    # Otherwise, the plot instance will be garbage collected after the initial draw
    # and you won't be able to move the plot elements around.
    # For related reasons, if you are using PyCharm, you have to execute the code in
    # a console (Alt+Shift+E).
    plot_instance = InteractiveGraph(graph_data)
    plt.show()

    # Create an editable plot, which is an interactive plot with the additions
    # that nodes and edges can be inserted or deleted, and labels and annotations
    # can be created, edited, or deleted as well.
    plot_instance = EditableGraph(graph_data)
    plt.show()

    # read the documentation for a full list of available arguments
    help(Graph)
    help(InteractiveGraph)
    help(EditableGraph)


Contributing & Support
----------------------

If you get stuck, please raise an issue on github_, or post a question
on stackoverflow_ using the `netgraph` tag. In either case, include
any relevant code and data in a `minimal, reproducible example`__. If
applicable, make a sketch of the desired result with pen and paper,
take a picture, and append it to the issue.

Bug reports are, of course, always welcome. Please make sure to
include the full error trace.

If you submit a pull request that fixes a bug or implements a
cool feature, I will probably worship the ground you walk on for the
rest of the week. Probably.

.. _github: https://github.com/paulbrodersen/netgraph
.. _stackoverflow: https://stackoverflow.com/
__ https://stackoverflow.com/help/minimal-reproducible-example


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Examples

   sphinx_gallery_output/index.rst

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: API Reference

   graph_classes.rst
   node_layout.rst
   edge_layout.rst
