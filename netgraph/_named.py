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
    DraggableArtists,
    DraggableGraph,
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


def _lateralize_arced_edge_paths(edge_paths, node_positions, above):
    """Ensure that edge paths are all either above or below the line passing through nodes."""
    for (source, target), path in edge_paths.items():
        p1 = node_positions[source]
        p2 = node_positions[target]
        if source != target:
            gradient, intercept = _get_gradient_and_intercept(p1, p2)
        else:
            gradient, intercept = 0., p1[1]
        mask = _is_above_line(path, gradient, intercept)
        if above:
            mask = np.invert(mask)
        path[mask] = _reflect_across_line(path[mask], gradient, intercept)
        edge_paths[(source, target)] = path
    return edge_paths


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
        edge_paths = _lateralize_arced_edge_paths(edge_paths, node_positions, self.above)
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


class DraggableArcDiagram(ArcDiagram, DraggableGraph):

    def __init__(self, *args, **kwargs):
        ArcDiagram.__init__(self, *args, **kwargs)
        DraggableArtists.__init__(self, self.node_artists.values())

        self._draggable_artist_to_node = dict(zip(self.node_artists.values(), self.node_artists.keys()))
        self._clickable_artists.extend(list(self.edge_artists.values()))
        self._selectable_artists.extend(list(self.edge_artists.values()))
        self._base_linewidth.update(dict([(artist, artist._lw_data) for artist in self.edge_artists.values()]))
        self._base_edgecolor.update(dict([(artist, artist.get_edgecolor()) for artist in self.edge_artists.values()]))

        # # trigger resize of labels when canvas size changes
        # self.fig.canvas.mpl_connect('resize_event', self._on_resize)

    def _update_node_positions(self, nodes, cursor_position):
        # cursor_position[1] = 0. # remove y-component to remain on the line
        for node in nodes:
            x, _ = cursor_position + self._offset[self.node_artists[node]]
            self.node_positions[node] = np.array([x, self.node_positions[node][1]])

    def _update_edges(self, edges):
        edge_paths = dict()
        edge_paths.update(self._update_arced_edge_paths([(source, target) for (source, target) in edges if source != target]))
        edge_paths.update(self._update_selfloop_paths([(source, target) for (source, target) in edges if source == target]))
        edge_paths = _lateralize_arced_edge_paths(edge_paths, self.node_positions, self.above)
        self.edge_paths.update(edge_paths)
        self._update_edge_artists(edge_paths)
