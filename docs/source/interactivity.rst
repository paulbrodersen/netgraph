.. _interactivity:

Interactive Graphs
==================

Algorithmically finding a visually pleasing graph layout is hard. This is demonstrated by the plethora of different algorithms in use (if graph layout was a solved problem, there would only be one algorithm). To ameliorate this problem, this module contains an :code:`InteractiveGraph` class, which allows node positions to be tweaked with the mouse after an initial draw.

  - Individual nodes and edges can be selected using the left-click.
  - Multiple nodes and or edges can be selected by holding :code:`control` while clicking, or by using the rectangle/window selector.
  - Selected plot elements can be dragged around by holding left-click on a selected artist.

.. image:: https://media.giphy.com/media/yEysQUUTndLT6mI9cN/giphy.gif
    :width: 400
    :align: center
    :alt: Demo of selecting, dragging, and hovering

.. code::

    import matplotlib.pyplot as plt
    import networkx as nx
    from netgraph import InteractiveGraph

    g = nx.house_x_graph()

    edge_color = dict()
    for ii, edge in enumerate(g.edges):
        edge_color[edge] = 'tab:gray' if ii%2 else 'tab:orange'

    node_color = dict()
    for node in g.nodes:
        node_color[node] = 'tab:red' if node%2 else 'tab:blue'

    plot_instance = InteractiveGraph(
        g, node_size=5, node_color=node_color,
        node_labels=True, node_label_offset=0.1, node_label_fontdict=dict(size=20),
        edge_color=edge_color, edge_width=2,
        arrows=True, ax=ax)

    plt.show()


There is also some experimental support for editing the graph elements interactively using the :code:`EditableGraph` class.

    - Pressing :code:`insert` or :code:`+` will add a new node to the graph.
    - Double clicking on two nodes successively will create an edge between them.
    - Pressing :code:`delete` or :code:`-` will remove selected nodes and edges.
    - Pressing :code:`@` will reverse the direction of selected edges.

When adding a new node, the properties of the last selected node will be used to style the node artist. Ditto for edges. If no node or edge has been previously selected, the first created node or edge artist will be used.

.. image:: https://media.giphy.com/media/TyiS2Pl1z9CFqYMYe7/giphy.gif
    :width: 400
    :align: center
    :alt: Demo of interactive editing

Finally, elements of the graph can be labeled and annotated. Labels remain always visible, whereas annotations can be toggled on and off by clicking on the corresponding node or edge.

    - To create or edit a node or edge label, select the node (or edge) artist, press the :code:`enter` key, and type.
    - To create or edit an annotation, select the node (or edge) artist, press :code:`alt + enter`, and type.
    - Terminate either action by pressing :code:`enter` or :code:`alt + enter` a second time.

.. image:: https://media.giphy.com/media/OofBM1xtwfSpK7DPSU/giphy.gif
    :width: 400
    :align: center
    :alt: Demo of interactive labelling

.. code::

    import matplotlib.pyplot as plt
    import networkx as nx
    from netgraph import EditableGraph

    g = nx.house_x_graph()

    edge_color = dict()
    for ii, (source, target) in enumerate(g.edges):
        edge_color[(source, target)] = 'tab:gray' if ii%2 else 'tab:orange'

    node_color = dict()
    for node in g.nodes:
        node_color[node] = 'tab:red' if node%2 else 'tab:blue'

    annotations = {
        4 : 'This is the representation of a node.',
        (0, 1) : dict(s='This is not a node.', color='red')
    }

    fig, ax = plt.subplots(figsize=(10, 10))

    plot_instance = EditableGraph(
        g, node_color=node_color, node_size=5,
        node_labels=True, node_label_offset=0.1, node_label_fontdict=dict(size=20),
        edge_color=edge_color, edge_width=2,
        annotations=annotations, annotation_fontdict = dict(color='blue', fontsize=15),
        arrows=True, ax=ax)

    plt.show()
