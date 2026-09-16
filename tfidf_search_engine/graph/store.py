from __future__ import annotations

from collections.abc import Iterable

from .edge import Edge
from .graph import Graph
from .node import Node


class GraphStore:
    """
    In-memory graph storage abstraction.

    GraphStore owns the lifecycle of the Graph object and provides
    persistence-oriented operations.

    A future implementation can replace this with:
    - PostgreSQL
    - Neo4j
    - another graph database
    - serialized storage
    """

    def __init__(self, graph: Graph | None = None) -> None:
        self._graph = graph if graph is not None else Graph()

    @property
    def graph(self) -> Graph:
        """Return the underlying graph."""
        return self._graph

    def add_node(self, node: Node) -> None:
        """Store a node."""
        self._graph.add_node(node)

    def add_edge(self, edge: Edge) -> None:
        """Store an edge."""
        self._graph.add_edge(edge)

    def get_node(self, node_id: str) -> Node | None:
        """Retrieve a node."""
        return self._graph.get_node(node_id)

    def nodes(self) -> Iterable[Node]:
        """Iterate through stored nodes."""
        return self._graph.nodes()

    def edges(self) -> Iterable[Edge]:
        """Iterate through stored edges."""
        return self._graph.edges()

    def delete_node(self, node_id: str) -> bool:
        """Delete a node and its relationships."""
        return self._graph.remove_node(node_id)

    def clear(self) -> None:
        """Clear the graph."""
        self._graph.clear()

    def node_count(self) -> int:
        """Return number of stored nodes."""
        return self._graph.node_count

    def edge_count(self) -> int:
        """Return number of stored edges."""
        return self._graph.edge_count

    