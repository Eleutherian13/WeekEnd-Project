from __future__ import annotations

from collections import deque
from collections.abc import Iterator

from .graph import Graph


class GraphTraversal:
    """
    Graph traversal algorithms.

    Traversal does not own graph data.
    It operates on an existing Graph.
    """

    def __init__(self, graph: Graph) -> None:
        self._graph = graph

    def bfs(
        self,
        start_node: str,
        max_depth: int | None = None,
    ) -> Iterator[str]:
        """
        Breadth-first traversal.

        Yields node IDs in BFS order.

        Parameters
        ----------
        start_node:
            Starting node.

        max_depth:
            Optional maximum traversal depth.
            0 means only the start node.
        """
        self._graph.require_node(start_node)

        if max_depth is not None:
            if not isinstance(max_depth, int):
                raise TypeError("max_depth must be an integer or None")

            if max_depth < 0:
                raise ValueError("max_depth must not be negative")

        queue: deque[tuple[str, int]] = deque(
            [(start_node, 0)]
        )

        visited: set[str] = {start_node}

        while queue:
            node_id, depth = queue.popleft()

            yield node_id

            if max_depth is not None and depth >= max_depth:
                continue

            for neighbor in self._graph.neighbors(node_id):
                if neighbor in visited:
                    continue

                visited.add(neighbor)
                queue.append((neighbor, depth + 1))

    def dfs(
        self,
        start_node: str,
        max_depth: int | None = None,
    ) -> Iterator[str]:
        """
        Depth-first traversal.

        Yields node IDs in DFS order.
        """
        self._graph.require_node(start_node)

        if max_depth is not None:
            if not isinstance(max_depth, int):
                raise TypeError("max_depth must be an integer or None")

            if max_depth < 0:
                raise ValueError("max_depth must not be negative")

        stack: list[tuple[str, int]] = [(start_node, 0)]
        visited: set[str] = {start_node}

        while stack:
            node_id, depth = stack.pop()

            yield node_id

            if max_depth is not None and depth >= max_depth:
                continue

            neighbors = self._graph.neighbors(node_id)

            for neighbor in reversed(neighbors):
                if neighbor in visited:
                    continue

                visited.add(neighbor)
                stack.append((neighbor, depth + 1))

    def k_hop_neighbors(
        self,
        start_node: str,
        k: int,
    ) -> set[str]:
        """
        Return all nodes reachable within k hops.

        The starting node itself is excluded.
        """
        if not isinstance(k, int):
            raise TypeError("k must be an integer")

        if k < 0:
            raise ValueError("k must not be negative")

        visited = set(self.bfs(start_node, max_depth=k))
        visited.discard(start_node)

        return visited