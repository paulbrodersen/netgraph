# netgraph

Python drawing utilities for publication quality plots of networks.


### Quickstart

Install with:

``` shell
pip install netgraph
```

Import module and plot with:

``` python
import numpy as np
import matplotlib.pyplot as plt
from netgraph import Graph, InteractiveGraph

# Several graph formats are supported:
graph_data = [(0, 1), (1, 2), (2, 0)] # edge list
# graph_data = [(0, 1, 0.2), (1, 2, -0.4), (2, 0, 0.7)] # edge list with weights
# graph_data = np.random.rand(10, 10) # full rank matrix
# graph_data = networkx.karate_club_graph() # networkx Graph/DiGraph objects
# graph_data = igraph.Graph.Famous('Zachary') # igraph Graph objects

# Create a non-interactive plot:
Graph(graph_data)
plt.show()

# Create an interactive plot.
# NOTE: you must retain a reference to the plot instance!
# Otherwise, the plot instance will be garbage collected after the initial draw
# and you won't be able to move the plot elements around.
plt.ion()
plot_instance = InteractiveGraph(graph_data)
plt.show()
```

## Reasons why you might want to use netgraph


### Better layouts

![Example visualisations](./figures/gallery_portrait.png)


### Interactive tweaking and data exploration

Algorithmically finding a visually pleasing graph layout is hard.
This is demonstrated by the plethora of different algorithms in use
(if graph layout was a solved problem, there would only be one
algorithm). To ameliorate this problem, this module contains an
`InteractiveGraph` class, which allows node positions to be tweaked
with the mouse after an initial draw.

The class `InteractiveGraph` also facilitates interactive data exploration.
When hovering over a node, the node and all its neighbours in the graph are highlighted.
When hovering over an edge, the edge and its source and target nodes are highlighted.

Apart from the labels, additional annotations can be passed in via the
`node_data` and `edge_data` keyword arguments. The visibility of these
annotations is toggled by clicking on the corresponding plot elements.

![Demo of InteractiveGraph](https://media.giphy.com/media/clrtFvPW1ITjtGyPIU/giphy.gif)


``` python
import matplotlib.pyplot as plt
import networkx as nx

from netgraph import InteractiveGraph

g = nx.house_x_graph()

node_data = {
    4 : dict(s = 'Additional annotations can be revealed\nby clicking on the corresponding plot element.', fontsize=20, backgroundcolor='white')
}
edge_data = {
    (0, 1) : dict(s='Clicking on the same plot element\na second time hides the annotation again.', fontsize=20, backgroundcolor='white')
}

fig, ax = plt.subplots(figsize=(10, 10))
plot_instance = InteractiveGraph(g, node_size=5, edge_width=3,
                                 node_labels=True, node_label_offset=0.08, node_label_fontdict=dict(size=20),
                                 node_data=node_data, edge_data=edge_data, ax=ax)
plt.show()
```


### Exquisite control over plot elements

High quality figures require fine control over plot elements.
To that end, all node artist and edge artist properties can be specified in three ways:

1. Using a single scalar or string that will be applied to all artists.

``` python
import matplotlib.pyplot as plt
from netgraph import Graph

edges = [(0, 1), (1, 1)]
Graph(edges, node_color='red', node_size=4.)
plt.show()
```

2. Using a dictionary mapping individual nodes or individual edges to a property:

``` python
import matplotlib.pyplot as plt
from netgraph import Graph

Graph([(0, 1), (1, 2), (2, 0)],
      edge_color={(0, 1) : 'g', (1, 2) : 'lightblue', (2, 0) : np.array([1, 0, 0])},
      node_size={0 : 20, 1 : 4.2, 2 : np.pi},
)
plt.show()
```

3. By directly manipulating the node and edge artists (which are simply matplotlib PathPatch artists):

``` python
import matplotlib.pyplot as plt; plt.ion()
from netgraph import Graph

fig, ax = plt.subplots()
g = Graph([(0, 1), (1, 2), (2, 0)], ax=ax)

# make some changes
g.edge_artists[(0, 1)].set_facecolor('red')
g.edge_artists[(1, 2)].set_facecolor('lightblue')

# force redraw to display changes
fig.canvas.draw()
```

Similarly, node and edge labels are just matplotlib text objects.
Their properties can also be specified using a single value that is applied to all of them:

``` python
import matplotlib.pyplot as plt
from netgraph import Graph

Graph([(0, 1)],
    node_size=20,
    node_labels={0 : 'Lorem', 1 : 'ipsum'},
    node_label_fontdict=dict(size=18, fontfamily='Arial', fontweight='bold'),
    edge_labels={(0, 1) : 'dolor sit'},
    # blue bounding box with red edge:
    edge_label_fontdict=dict(bbox=dict(boxstyle='round',
                                       ec=(1.0, 0.0, 0.0),
                                       fc=(0.5, 0.5, 1.0))),
)
plt.show()
```

Alternatively, their properties can be manipulated individually after an initial draw:

``` python
import matplotlib.pyplot as plt
from netgraph import Graph

fig, ax = plt.subplots()
g = Graph([(0, 1)],
    node_size=20,
    node_labels={0 : 'Lorem', 1 : 'ipsum'},
    edge_labels={(0, 1) : 'dolor sit'},
    ax=ax
)

# make some changes
g.node_label_artists[1].set_color('hotpink')
g.edge_label_artists[(0, 1)].set_style('italic')

# force redraw to display changes
fig.canvas.draw()
plt.show()
```

### Consistent length units

Existing drawing routines for networks in python (networkx, igraph)
use fundamentally different length units for different plot elements.
For example, networkx uses data units to specify node positions but
display units for node sizes. This makes it difficult to judge the
relative sizes of plot elements a priori. Also, layouts cannot be
exactly reproduced on different computers, if their display sizes
differ.

This module amends these issues by having a single reference frame
that derives from the data. Specifically, node positions and edge
paths are specified in data units, and node sizes and edge widths are
given in 1/100 of data units (i.e. a node with `node_size=2` has a
radius of 0.02 in data units). Rescaling by 1/100 makes the node sizes
and edge widths more comparable to typical node sizes in igraph and
networkx.


### Compatibility with igraph and networkx

Many people that analyse networks in python use several network analysis libraries, e.g. igraph and networkx.
To facilitate interoperability, various network formats are supported:

1. Edge lists:

   Iterable of (source, target) or (source, target, weight) tuples,
   or equivalent (m, 2) or (m, 3) ndarray.

2. Adjacency matrices:

   Full-rank (n, n) ndarray, where n corresponds to the number of nodes.
   The absence of a connection is indicated by a zero.

3. igraph.Graph or networkx.Graph objects


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
being said, the blue little notification dot on github is surprisingly
effective at getting my attention. So please just raise an issue.
