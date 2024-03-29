
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "sphinx_gallery_output/plot_08_dot_layout.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_sphinx_gallery_output_plot_08_dot_layout.py>`
        to download the full example code

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_sphinx_gallery_output_plot_08_dot_layout.py:


Dot and radial node layouts
===========================

Plot a tree or other directed, acyclic graph with the :code:`'dot'` or :code:`'radial'` node layout.
Netgraph uses an implementation of the Sugiyama algorithm provided by the grandalf_ library
(and thus does not require Graphviz to be installed).

.. _grandalf: https://github.com/bdcht/grandalf

.. GENERATED FROM PYTHON SOURCE LINES 12-42



.. image-sg:: /sphinx_gallery_output/images/sphx_glr_plot_08_dot_layout_001.png
   :alt: plot 08 dot layout
   :srcset: /sphinx_gallery_output/images/sphx_glr_plot_08_dot_layout_001.png
   :class: sphx-glr-single-img





.. code-block:: default


    import matplotlib.pyplot as plt
    import networkx as nx

    from netgraph import Graph

    unbalanced_tree = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (2, 6),
        (3, 7),
        (3, 8),
        (4, 9),
        (4, 10),
        (4, 11),
        (5, 12),
        (5, 13),
        (5, 14),
        (5, 15)
    ]

    balanced_tree = nx.balanced_tree(3, 3)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    Graph(unbalanced_tree, node_layout='dot', ax=ax1)
    Graph(balanced_tree, node_layout='radial', ax=ax2)
    plt.show()


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.366 seconds)


.. _sphx_glr_download_sphinx_gallery_output_plot_08_dot_layout.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_08_dot_layout.py <plot_08_dot_layout.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_08_dot_layout.ipynb <plot_08_dot_layout.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
