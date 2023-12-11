# Netgraph

*Publication-quality Network Visualisations in Python*

[![Downloads](https://pepy.tech/badge/netgraph)](https://pepy.tech/project/netgraph) [![DOI](https://zenodo.org/badge/70009270.svg)](https://zenodo.org/badge/latestdoi/70009270) [![DOI](https://joss.theoj.org/papers/10.21105/joss.05372/status.svg)](https://doi.org/10.21105/joss.05372)

Netgraph is a Python library that aims to complement existing network analysis libraries such as such as [networkx](https://networkx.org/), [igraph](https://igraph.org/), and [graph-tool](https://graph-tool.skewed.de/) with publication-quality visualisations within the Python ecosystem. To facilitate a seamless integration, Netgraph supports a variety of input formats, including networkx, igraph, and graph-tool `Graph` objects. Netgraph implements numerous node layout algorithms and several edge routing routines. Uniquely among Python alternatives, it handles networks with multiple components gracefully (which otherwise break most node layout routines), and it post-processes the output of the node layout and edge routing algorithms with several heuristics to increase the interpretability of the visualisation (reduction of overlaps between nodes, edges, and labels; edge crossing minimisation and edge unbundling where applicable). The highly customisable plots are created using [Matplotlib](https://matplotlib.org/), and the resulting Matplotlib objects are exposed in an easily queryable format such that they can be further manipulated and/or animated using standard Matplotlib syntax. Finally, Netgraph also supports interactive changes: with the `InteractiveGraph` class, nodes and edges can be positioned using the mouse, and the `EditableGraph` class additionally supports insertion and deletion of nodes and edges as well as their (re-)labelling through standard text-entry.

## Installation

Install the current release of `netgraph` from PyPI:

``` shell
pip install netgraph
```

If you are using (Ana-)conda (or mamba), you can also obtain Netgraph from conda-forge:

``` shell
conda install -c conda-forge netgraph
```

## Documentation

Numerous tutorials, code examples, and a complete documentation of the API can be found on [ReadTheDocs](https://netgraph.readthedocs.io/en/latest/index.html).

## Quickstart

``` python
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

# Netgraph uses Matplotlib for creating the visualisation.
# Node and edge artistis are derived from `matplotlib.patches.PathPatch`.
# Node and edge labels are `matplotlib.text.Text` instances.
# Standard matplotlib syntax applies.
fig, ax = plt.subplots(figsize=(5,4))
plot_instance = Graph([(0, 1)], node_labels=True, edge_labels=True, ax=ax)
plot_instance.node_artists[0].set_alpha(0.2)
plot_instance.edge_artists[(0, 1)].set_facecolor('red')
plot_instance.edge_label_artists[(0, 1)].set_style('italic')
plot_instance.node_label_artists[1].set_size(10)
ax.set_title("This is my fancy title.")
ax.set_facecolor('honeydew') # change background color
fig.canvas.draw() # force redraw to display changes
fig.savefig('test.pdf', dpi=300)
plt.show()

# Read the documentation for a full list of available arguments:
help(Graph)
help(InteractiveGraph)
help(EditableGraph)
```

## Examples


![Example visualisations](./figures/gallery_portrait.png)


## Citing Netgraph

If you use Netgraph in a scientific publication, I would appreciate citations to the following paper:

Brodersen, P. J. N., (2023). Netgraph: Publication-quality Network Visualisations in Python. Journal of Open Source Software, 8(87), 5372, https://doi.org/10.21105/joss.05372

Bibtex entry:

```bibtex
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
```

## Recent changes

- 4.13.1 Improved initialization of `k` parameter in `get_fruchterman_reingold_layout` (issue #79).
- 4.13.0 Wrote an article on Netgraph for the Journal of Open Source Software (latest draft in /publication).
- 4.12.12 Expanded the documentation to cover installation of optional dependencies, automated testing, and troubleshooting issues with Matplotlib event handling (issue #69).
- 4.12.11 Mitigated a bug in `EditableGraph` that occurred when deleting a node while hovering over an edge incident to that node (issue #66).
- 4.12.10 Fixed a bug with automatic node label rescaling if the node label fontsize was specified using the `fontsize` keyword argument (instead of just `size`).
- 4.12.9 Fixed a bug that occurred when the distance argument to `_shorten_line_by` was equal or smaller than zero.
- 4.12.8 Fixed a bug that occurred with recent numpy versions when using multi-partite or shell layouts with un-equal numbers of nodes in each layer (issue #65).
- 4.12.7 Fixed a bug that occurred with recent Matplotlib versions when using the rectangle selector in `InteractiveGraph`.
- 4.12.6 Added support for graphs with nodes but no edges to `EditableGraph` (issue #62).
- 4.12.5 Added support for empty graphs in `EditableGraph` (issue #62).
- 4.12.4 Turned off clipping of self-loop paths.
- 4.12.3 Bugfix: stopped overwriting `step` parameter in `get_community_layout`.
- 4.12.2 Improved node positions rescaling for some layouts & standardised node position padding across all layouts.
- 4.12.1 Fixed a bug/deprecation issue that occurred with new versions of Matplotlib if an axis was provided (thanks @speedsmith!).
- 4.12.0 Implemented the geometric node layout, which infers node positions from given edge lengths.
- 4.11.8 Plotting of graphs with a single node is now actually supported. Thanks @Alexander-Wilms.
- 4.11.7 Removed instances of (deprecated) np.float / np.int types (issue #58).
- 4.11.6 Added documentation on hyperlinks (issue #56).
- 4.11.5 Resolved a SyntaxWarning.
- 4.11.4 Fixed a bug that occurred when using edge annotations.
- 4.11.3 Cleaned build directory and removed several outdated files.
- 4.11.2 Fixed a bug that prevented changing the axis background colour.
- 4.11.1 Resolved warnings during build process.
- 4.11.0 Switched from setup.py based installation to pyproject.toml/wheels.
- 4.10.4 Added support for custom graph classes that inherit from networkx/igraph/graph-tool base Graph classes (issue #53).
- 4.10.3 Improved the error message for incomplete iterable arguments (issue #55).
- 4.10.2 Fixed issue #48 (again, sorry).
- 4.10.1 Implemented flag that controls bundling of parallel edges in the curved edge layout (`bundle_parallel_edges`).
- 4.10.0 Implemented grid-mode for the InteractiveGraph class to facilitate alignment of nodes (toggle with 'g').
- 4.9.7 Fixed a bug introduced in version 4.9.5 in the computation of repulsive forces in the spring layout (i.e. the default layout).
- 4.9.6 Fixed issue #51, which occurred in the community node layout if a node had no connections to other nodes in its community.
- 4.9.5 Improved the routine that reduces node overlaps in the spring and community node layouts.
- 4.9.4 Fixed a bug that occurred in get_curved_edges if the k parameter was not set explicitly.
- 4.9.3 Improved placement of nodes in instances where all nodes are aligned within any one dimension.
- 4.9.2 Fixed an issue that occurred when using the dot node layout for a graphs with multiple components.
- 4.9.1 Fixed issue #48, which occurred when computing a curved edge layout for graphs with multiple components. Also improved edge routing slightly.
- 4.9.0 Implemented a layered and a shell layout for multi-partite graphs.
- 4.8.2 Fixed issue #45, which prevented a warning to be raised when trying to plot networkx.MultiGraph instances.
- 4.8.1 Fixed issue #44, that occurred if a single node was plotted with the Sugiyama layout (thanks @wilkeber).
- 4.8.0 Refined community node layout. Communities are rotated w.r.t. each other to reduce the length of edges between them.
- 4.7.1 Fixed issue #41, which occurred when computing the community layout for communities without within-community edges.
- 4.7.0 Implemented a radial tree node layout.
- 4.6.0 Support drawing of graph_tool.Graph objects.
- 4.5.0 Support custom mappings for mouse-over highlight events (issue #39).
- 4.4.1 Corrected imports for testing _arcdiagram.py.
- 4.4.0 Added bipartite node layouts.
- 4.3.0 Added the ArcDiagram class and interactive/editable variants.
- 4.2.4 Plotting of graphs with a single node is now supported.
- 4.2.3 Fixed a bug that occurred when using the community layout with at least one community containing a single node.
- 4.2.2 Fixed a bug that occurred every time an InteractiveGraph was initialised without tabular annotations.
- 4.2.1 Added online documentation.

## Help, I don't know how to do ...!

Please raise an issue. Include any relevant code and data in a
[minimal, reproducible
example](https://stackoverflow.com/help/minimal-reproducible-example).
If applicable, make a sketch of the desired result with pen and paper,
take a picture, and append it to the issue.

Bug reports are, of course, always welcome. Please make sure to
include the full error trace.

If you submit a pull request that fixes a bug or implements a
cool feature, I will probably worship the ground you walk on for the
rest of the week. Probably.

Finally, if you do email me, please be very patient. I rarely check
the email account linked to my open source code, so I probably will
not see your emails for several weeks, potentially longer. Also, I have a
job that I love and that pays my bills, and thus takes priority. That
being said, the blue little notification dot on GitHub is surprisingly
effective at getting my attention. So please just raise an issue.
