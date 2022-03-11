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
from ._utils import (
    _get_gradient_and_intercept,
    _is_above_line,
    _reflect_across_line
)


class BaseArcDiagram(BaseGraph):

    def __init__(self, edges, nodes=None, node_layout='linear', node_order=None, above=True, *args, **kwargs):
        self.above = above
        kwargs.setdefault('node_layout_kwargs', dict())
        kwargs['node_layout_kwargs'].setdefault('node_order', node_order)
        if node_order:
            kwargs['node_layout_kwargs'].setdefault('reduce_edge_crossings', False)
        else:
            kwargs['node_layout_kwargs'].setdefault('reduce_edge_crossings', True)
        super().__init__(edges, nodes=nodes, node_layout=node_layout, edge_layout='arc', *args, **kwargs)

    def _get_edge_paths(self, edges, node_positions, edge_layout, edge_layout_kwargs):
        edge_paths = super()._get_edge_paths(edges, node_positions, edge_layout, edge_layout_kwargs)

        for (source, target), path in edge_paths.items():
            p1 = self.node_positions[source]
            p2 = self.node_positions[target]
            gradient, intercept = _get_gradient_and_intercept(p1, p2)
            mask = _is_above_line(path, gradient, intercept)
            if self.above:
                mask = np.invert(mask)
            path[mask] = _reflect_across_line(path[mask], gradient, intercept)
            edge_paths[(source, target)] = path

        return edge_paths


class ArcDiagram(BaseArcDiagram, Graph):

    def __init__(self, graph, node_layout='linear', node_order=None, above=True, *args, **kwargs):
        self.above = above
        kwargs.setdefault('node_layout_kwargs', dict())
        kwargs['node_layout_kwargs'].setdefault('node_order', node_order)
        if node_order:
            kwargs['node_layout_kwargs'].setdefault('reduce_edge_crossings', False)
        else:
            kwargs['node_layout_kwargs'].setdefault('reduce_edge_crossings', True)
        Graph.__init__(self, graph, node_layout=node_layout, edge_layout='arc', *args, **kwargs)
