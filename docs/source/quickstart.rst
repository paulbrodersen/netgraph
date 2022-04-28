.. _quickstart:

Quickstart
==========

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

    # 6) graph_tool.Graph objects
    import graph_tool.collection
    graph_data = graph_tool.collection.data["karate"]

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
