
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "sphinx_gallery_output/plot_14_bipartite_layout.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_sphinx_gallery_output_plot_14_bipartite_layout.py>`
        to download the full example code

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_sphinx_gallery_output_plot_14_bipartite_layout.py:


Bipartite node layout
=====================


.. GENERATED FROM PYTHON SOURCE LINES 9-11

By default, nodes are partitioned into two subsets using a two-coloring of the graph.
The median heuristic proposed in Eades & Wormald (1994) is used to reduce edge crossings.

.. GENERATED FROM PYTHON SOURCE LINES 11-28

.. code-block:: default


    import matplotlib.pyplot as plt

    from netgraph import Graph

    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (5, 6)
    ]

    Graph(edges, node_layout='bipartite', node_labels=True)

    plt.show()




.. image-sg:: /sphinx_gallery_output/images/sphx_glr_plot_14_bipartite_layout_001.png
   :alt: plot 14 bipartite layout
   :srcset: /sphinx_gallery_output/images/sphx_glr_plot_14_bipartite_layout_001.png
   :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /home/paul/src/netgraph/netgraph/_node_layout.py:1026: UserWarning: The graph consistst of multiple components, and hence the partitioning into two subsets/layers is ambiguous!
    Use the `subsets` argument to explicitly specify the desired partitioning.
      warnings.warn(msg)




.. GENERATED FROM PYTHON SOURCE LINES 29-32

The partitions can also be made explicit using the :code:`subsets` argument.
In multi-component bipartite graphs, multiple two-colorings are possible,
such that explicit specification of the subsets may be necessary to achieve the desired partitioning of nodes.

.. GENERATED FROM PYTHON SOURCE LINES 32-49

.. code-block:: default


    import matplotlib.pyplot as plt

    from netgraph import Graph

    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (5, 6)
    ]

    Graph(edges, node_layout='bipartite', node_layout_kwargs=dict(subsets=[(0, 2, 4, 6), (1, 3, 5)]), node_labels=True)

    plt.show()




.. image-sg:: /sphinx_gallery_output/images/sphx_glr_plot_14_bipartite_layout_002.png
   :alt: plot 14 bipartite layout
   :srcset: /sphinx_gallery_output/images/sphx_glr_plot_14_bipartite_layout_002.png
   :class: sphx-glr-single-img





.. GENERATED FROM PYTHON SOURCE LINES 50-52

To change the layout from the left-right orientation to a bottom-up orientation,
call the layout function directly and swap x and y coordinates of the node positions.

.. GENERATED FROM PYTHON SOURCE LINES 52-71

.. code-block:: default


    import matplotlib.pyplot as plt

    from netgraph import Graph, get_bipartite_layout

    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (5, 6)
    ]

    node_positions = get_bipartite_layout(edges, subsets=[(0, 2, 4, 6), (1, 3, 5)])
    node_positions = {node : (x, y) for node, (y, x) in node_positions.items()}

    Graph(edges, node_layout=node_positions, node_labels=True)

    plt.show()



.. image-sg:: /sphinx_gallery_output/images/sphx_glr_plot_14_bipartite_layout_003.png
   :alt: plot 14 bipartite layout
   :srcset: /sphinx_gallery_output/images/sphx_glr_plot_14_bipartite_layout_003.png
   :class: sphx-glr-single-img






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.362 seconds)


.. _sphx_glr_download_sphinx_gallery_output_plot_14_bipartite_layout.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_14_bipartite_layout.py <plot_14_bipartite_layout.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_14_bipartite_layout.ipynb <plot_14_bipartite_layout.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
