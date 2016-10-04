# netgraph
Fork of networkx drawing utilities for publication quality plots of networks

## Summary:
----
Module to plot weighted, directed graphs of medium size (10-100 nodes).
Unweighted, undirected graphs will look perfectly fine, too, but this module
might be overkill for such a use case.

## Raison d'etre:
---
Existing draw routines for networks/graphs in python use fundamentally different
length units for different plot elements. This makes it hard to
    - provide a consistent layout for different axis / figure dimensions, and
    - judge the relative sizes of elements a priori.
This module amends these issues (while sacrificing speed).

## Example:
---
```python
import numpy as np
import matplotlib.pyplot as plt
import netgraph

# construct sparse, directed, weighted graph
# with positive and negative edges
n = 20
w = np.random.randn(n,n)
p = 0.2
c = np.random.rand(n,n) <= p
w[~c] = np.nan

# plot
netgraph.draw(w)
plt.show()
```
