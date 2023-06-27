---
title: 'Netgraph: Publication-quality Network Visualisations in Python'
tags:
  - Python
  - graph
  - network
  - visualisation
  - visualization
  - matplotlib
  - networkx
  - igraph
  - graph-tool
authors:
  - name: Paul J. N. Brodersen
    orcid: 0000-0001-5216-7863
    affiliation: 1
affiliations:
  - name: Department of Pharmacology, University of Oxford, United Kingdom
    index: 1
date: 16 March 2023
bibliography: paper.bib
---

# Statement of need

The empirical study and scholarly analysis of networks has increased manyfold in recent decades, fuelled by the new prominence of network structures in our lives (the web, social networks, artificial neural networks, ecological networks, etc.) and the data available on them. While there are several comprehensive Python libraries for network analysis such as NetworkX [@Hagberg:2008], igraph [@Csardi:2006], and graph-tool [@Peixoto:2014], their inbuilt visualisation capabilities lag behind specialised software solutions such as Graphviz [@Ellson:2002], Cytoscape [@Shannon:2003], or Gephi [@Bastian:2009]. However, although Python bindings for these applications exist in the form of PyGraphviz, py4cytoscape, and GephiStreamer, respectively, their outputs are not manipulable Python objects, which restricts customisation, limits their extensibility, and prevents a seamless integration within a wider Python application.

# Summary

Netgraph is a Python library that aims to complement the existing network analysis libraries with publication quality visualisations within the Python ecosystem. To facilitate a seamless integration, Netgraph supports a variety of input formats, including NetworkX, igraph, and graph-tool Graph objects. At the time of writing, Netgraph provides the following node layout algorithms:

- the Fruchterman-Reingold algorithm a.k.a. the "spring" layout,
- the Sugiyama algorithm a.k.a. the "dot" layout for directed, acyclic graphs,
- a radial tree layout for directed, acyclic graphs,
- a circular node layout (with optional edge crossing reduction),
- a bipartite node layout for bipartite graphs (with optional edge crossing reduction),
- a layered node layout for multipartite graphs (with optional edge crossing reduction),
- a shell layout for multipartite graphs (with optional edge crossing reduction),
- a community node layout for modular graphs, and
- a "geometric" node layout for graphs with defined edge lengths but unknown node positions.

Additionally, links or edges between the nodes can be straight, curved (avoiding collisions with other nodes and edges), or bundled.
However, new layout routines are added regularly to Netgraph; for an up-to-date list, consult the online documentation [here](https://netgraph.readthedocs.io/en/latest/node_layout.html).

Uniquely among Python alternatives, Netgraph handles networks with multiple components gracefully (which otherwise break most node layout routines), and it post-processes the output of the node layout and edge routing algorithms with several heuristics to increase the interpretability of the visualisation (reduction of overlaps between nodes, edges, and labels; edge crossing minimisation and edge unbundling where applicable). The highly customisable plots are created using Matplotlib [@Hunter:2007], a popular Python plotting library, and the resulting Matplotlib objects are exposed in an easily queryable format such that they can be further manipulated and/or animated using standard Matplotlib syntax. The visualisations can also be altered interactively: nodes and edges can be added on-the-fly through hotkeys, positioned using the mouse, and labelled through standard text-entry.

Netgraph is licensed under the General Public License version 3 (GPLv3). The repository is hosted on [GitHub](https://github.com/paulbrodersen/netgraph), and distributed via PyPI and conda-forge. It includes an extensive automated test suite that can be executed using pytest. The comprehensive documentation -- including a complete API reference as well as numerous examples and tutorials -- is hosted on [ReadTheDocs](https://netgraph.readthedocs.io).

# Figures

![Netgraph's key distinguishing features](gallery_portrait.png){width=90%}

# Acknowledgements

We thank GitHub users adleris, Allan L. R. Hansen, chenghuzi, Hamed Mohammadpour, and Pablo for contributing various bug fixes.

# References
