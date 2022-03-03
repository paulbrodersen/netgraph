#!/usr/bin/env python
"""
For some special (and often named) network visualisations, the node
placement and edge routing are inseparable. For example, the placement
of nodes in a line in arc-diagrams necessitates the representation of
edges as arcs. These special cases are implemented here.
"""

import numpy as np

from ._main import (
    BaseGraph,
    Graph,
    InteractiveGraph,
)
from ._interactive_variants import (
    EditableGraph,
)
from ._node_layout import (
    get_linear_layout,
)
from ._edge_layout import (
    get_arced_edge_paths,
    get_selfloop_paths,
)

class BaseArcDiagram(BaseGraph):

    def __init__(self, edges, nodes=None,
                 node_layout='linear', edge_layout='arc',
                 arc_above=True, origin=(0, -0.5), scale=(1, 1),
                 *args, **kwargs):

        self.arc_above = arc_above
        super().__init__(edges, nodes=nodes,
                         node_layout=node_layout, edge_layout=edge_layout,
                         origin=origin, scale=scale,
                         *args, **kwargs)

    def _get_edge_paths(self, *args, **kwargs):
        edge_paths = super()._get_edge_paths(*args, **kwargs)
        if self.arc_above:
            edge_paths = {edge : np.c_[path[:,0],  np.abs(path[:,1])] for edge, path in edge_paths.items()}
        else:
            edge_paths = {edge : np.c_[path[:,0], -np.abs(path[:,1])] for edge, path in edge_paths.items()}
        return edge_paths
