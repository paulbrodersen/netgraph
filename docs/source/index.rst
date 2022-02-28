.. Netgraph documentation master file, created by
   sphinx-quickstart on Mon Feb 21 12:02:02 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Netgraph
========

Netgraph is a python library for creating publication quality plots of networks.
It is compatible with a variety of network data formats, including :code:`networkx` and :code:`igraph` :code:`Graph` objects.


Installation
------------

.. code-block:: shell

    pip install netgraph


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
.. _StackOverflow: https://stackoverflow.com/
__ https://stackoverflow.com/help/minimal-reproducible-example


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Documentation

   installation.rst
   quickstart.rst
   sphinx_gallery_output/index.rst

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: API Reference

   graph_classes.rst
   node_layout.rst
   edge_layout.rst
