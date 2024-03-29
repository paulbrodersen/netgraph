
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "sphinx_gallery_output/plot_13_arcdiagram.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_sphinx_gallery_output_plot_13_arcdiagram.py>`
        to download the full example code

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_sphinx_gallery_output_plot_13_arcdiagram.py:


Arc Diagrams
============

.. GENERATED FROM PYTHON SOURCE LINES 6-35

.. code-block:: default


    import matplotlib.pyplot as plt
    import networkx as nx

    from netgraph import ArcDiagram

    # Create a modular graph.
    partition_sizes = [5, 6, 7]
    g = nx.random_partition_graph(partition_sizes, 1, 0.1)

    # Create a dictionary that maps nodes to the community they belong to,
    # and set the node colors accordingly.
    node_to_community = dict()
    node = 0
    for community_id, size in enumerate(partition_sizes):
        for _ in range(size):
            node_to_community[node] = community_id
            node += 1

    community_to_color = {
        0 : 'tab:blue',
        1 : 'tab:orange',
        2 : 'tab:green',
    }
    node_color = {node: community_to_color[community_id] for node, community_id in node_to_community.items()}

    ArcDiagram(g, node_size=1, node_color=node_color, node_edge_width=0, edge_alpha=1., edge_width=0.1)
    plt.show()




.. image-sg:: /sphinx_gallery_output/images/sphx_glr_plot_13_arcdiagram_001.png
   :alt: plot 13 arcdiagram
   :srcset: /sphinx_gallery_output/images/sphx_glr_plot_13_arcdiagram_001.png
   :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /home/paul/src/netgraph/netgraph/_node_layout.py:847: UserWarning: Maximum number of iterations reached. Aborting further node layout optimisations.
      warnings.warn("Maximum number of iterations reached. Aborting further node layout optimisations.")




.. GENERATED FROM PYTHON SOURCE LINES 36-40

By default, ArcDiagram optimises the node order such that the number of edge crossings is minimised.
For larger graphs, this process can take a long time.
The node order can be set explicitly using the :code:`node_order` argument.
In this case, no optimisation is attempted.

.. GENERATED FROM PYTHON SOURCE LINES 40-42

.. code-block:: default


    ArcDiagram(g, node_order=range(len(g)), node_size=1, node_color=node_color, node_edge_width=0, edge_alpha=1., edge_width=0.1)



.. image-sg:: /sphinx_gallery_output/images/sphx_glr_plot_13_arcdiagram_002.png
   :alt: plot 13 arcdiagram
   :srcset: /sphinx_gallery_output/images/sphx_glr_plot_13_arcdiagram_002.png
   :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    <netgraph._arcdiagram.ArcDiagram object at 0x7f034f543d30>




.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 1 minutes  0.919 seconds)


.. _sphx_glr_download_sphinx_gallery_output_plot_13_arcdiagram.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_13_arcdiagram.py <plot_13_arcdiagram.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_13_arcdiagram.ipynb <plot_13_arcdiagram.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
