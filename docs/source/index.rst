.. Netgraph documentation master file, created by
   sphinx-quickstart on Mon Feb 21 12:02:02 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Netgraph
========

*Publication-quality network visualisations in python*

.. image:: https://pepy.tech/badge/netgraph

Netgraph is a Python library that aims to complement existing network analysis libraries such as such as networkx_, igraph_, and graph-tool_ with publication-quality visualisations within the Python ecosystem. To facilitate a seamless integration, Netgraph supports a variety of input formats, including networkx, igraph, and graph-tool :code:`Graph` objects. Netgraph implements numerous node layout algorithms and several edge routing routines. Uniquely among Python alternatives, it handles networks with multiple components gracefully (which otherwise break most node layout routines), and it post-processes the output of the node layout and edge routing algorithms with several heuristics to increase the interpretability of the visualisation (reduction of overlaps between nodes, edges, and labels; edge crossing minimisation and edge unbundling where applicable). The highly customisable plots are created using Matplotlib_, and the resulting Matplotlib objects are exposed in an easily queryable format such that they can be further manipulated and/or animated using standard Matplotlib syntax. The visualisations can also be altered interactively: nodes and edges can be added on-the-fly through hotkeys, positioned using the mouse, and (re-)labelled through standard text-entry.

.. _networkx: https://networkx.org/
.. _igraph: https://igraph.org/
.. _graph-tool: https://graph-tool.skewed.de/
.. _Matplotlib: https://matplotlib.org/


Citing Netgraph
---------------

If you use Netgraph in a scientific publication, I would appreciate citations to the following paper:


Brodersen, P. J. N., (2023). Netgraph: Publication-quality Network Visualisations in Python. Journal of Open Source Software, 8(87), 5372, https://doi.org/10.21105/joss.05372


Bibtex entry:

.. code-block::

    @article{Brodersen2023,
        doi     = {10.21105/joss.05372},
        url     = {https://doi.org/10.21105/joss.05372},
        year    = {2023}, publisher = {The Open Journal},
        volume  = {8},
        number  = {87},
        pages   = {5372},
        author  = {Paul J. N. Brodersen},
        title   = {Netgraph: Publication-quality Network Visualisations in Python},
        journal = {Journal of Open Source Software},
    }


Contributing & Support
----------------------

If you get stuck, please raise an issue on GitHub_, or post a question
on StackOverflow_ using the :code:`netgraph` tag. In either case, include
any relevant code and data in a `minimal, reproducible example`__. If
applicable, make a sketch of the desired result with pen and paper,
take a picture, and append it to the issue.

Bug reports are, of course, always welcome. Please make sure to
include the full error trace.

If you submit a pull request that fixes a bug or implements a
cool feature, I will probably worship the ground you walk on for the
rest of the week. Probably.

.. _GitHub: https://github.com/paulbrodersen/netgraph
.. _StackOverflow: https://stackoverflow.com/questions/tagged/netgraph
__ https://stackoverflow.com/help/minimal-reproducible-example


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Documentation

   installation.rst
   quickstart.rst
   sphinx_gallery_output/index.rst
   interactivity.rst
   sphinx_gallery_animations/index.rst

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: API Reference

   graph_classes.rst
   arcdiagram_classes.rst
   node_layout.rst
   edge_layout.rst
