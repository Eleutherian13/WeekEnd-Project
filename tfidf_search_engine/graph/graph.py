from __future__ import annotations

from collections.abc import Iterable, Iterator
from .adjacency import AdjacencyList
from .node import Node
from .edge import Edge

class Graph : 

    """ directed graph contains nodes and edges 
    graph is resp for node life cycle , edge life cycle , graph invariant , delegating connectivity  storage to Adjacency """

    def __init__(self ) -> None : 

        self._nodes : dict[str , Node] = {}

        self.adjacency = AdjacencyList()


    def add_node(self, node: Node) -> None:
        if not isinstance(node, Node):
            raise TypeError("node must be a Node")

        if node.node_id in self._nodes : 
            raise ValueError(f"node with id {node.node_id} already exists")

        self._nodes[node.node_id] = node 

    add_nodes = add_node

    def get_node(self, node_id: str) -> Node | None:
        """Return a node by ID or None when it is absent."""
        return self._nodes.get(node_id)

    def require_node(self, node_id: str) -> Node:
        """Return a node or raise KeyError."""
        try:
            return self._nodes[node_id]
        except KeyError:
            raise KeyError(f"Unknown node: {node_id}") from None


    def remove_node(self , node_id : str ) -> bool : 
        if node_id not in self._nodes : 
            return False 

        del self._nodes[node_id]

        # remove all edges connected to this node
        for edge in list(self.adjacency.outgoing_edges(node_id)):
            self.adjacency.remove_edge(edge)

        return True

    def has_node(self , node_id : str) -> bool : 
        return node_id in self._nodes

    def nodes(self ) -> Iterator[Node] : 

        return iter(self._nodes.values())

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------

    def add_edge(self, edge: Edge) -> None:
        """
        Add an edge.

        Both source and target nodes must already exist.
        """
        if not isinstance(edge, Edge):
            raise TypeError("edge must be an Edge")

        if not self.has_node(edge.source_node):
            raise KeyError(
                f"Source node does not exist: {edge.source_node}"
            )

        if not self.has_node(edge.target_node):
            raise KeyError(
                f"Target node does not exist: {edge.target_node}"
            )

        self.adjacency.add(edge)

    def remove_edge(self, edge: Edge) -> bool:
        """Remove an exact edge."""
        return self.adjacency.remove_edge(edge)

    def has_edge(self, source: str, target: str) -> bool:
        """Return whether an edge exists between two nodes."""
        return self.adjacency.has_edge(source, target)

    def neighbors(self, node_id: str) -> tuple[str, ...]:
        """Return neighboring node IDs."""
        self.require_node(node_id)
        return self.adjacency.neighbors(node_id)

    def outgoing_edges(self, node_id: str) -> tuple[Edge, ...]:
        """Return outgoing edges."""
        self.require_node(node_id)
        return self.adjacency.outgoing_edges(node_id)

    def edges(self) -> Iterable[Edge]:
        """Iterate through all edges."""
        return self.adjacency.edges()
    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self.adjacency)

    def clear(self) -> None:
        """Remove all nodes and edges."""
        self._nodes.clear()
        self.adjacency.clear()