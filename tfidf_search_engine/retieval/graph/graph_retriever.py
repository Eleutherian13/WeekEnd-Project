from __future__ import annotations

from collections.abc import Sequence

from graph.graph import Graph
from graph.traversal import GraphTraversal
from retieval.lexical.retriever import RetrievalResult
from retieval.graph.retriever import GraphRetriever


class SimpleGraphRetriever(GraphRetriever):
    """
    Concrete graph-based retriever using graph traversal.

    Retrieval strategy
    ------------------
    1. Use supplied graph nodes as retrieval starting points.
    2. Traverse the graph up to a configurable number of hops.
    3. Convert reached document nodes into retrieval results.
    4. Score documents according to their graph distance.
    5. Return the highest-scoring documents.

    This implementation intentionally does not perform semantic
    query-to-node matching. The caller supplies starting nodes.

    That keeps graph traversal independent from query understanding.
    A more advanced implementation can later add entity linking,
    vector-based node matching, or query planning.
    """

    def __init__(
        self,
        graph: Graph,
        start_nodes: Sequence[str] | None = None,
        max_hops: int = 2,
        document_node_type: str = "document",
    ) -> None:
        if not isinstance(graph, Graph):
            raise TypeError("graph must be a Graph")

        if not isinstance(max_hops, int):
            raise TypeError("max_hops must be an integer")

        if isinstance(max_hops, bool):
            raise TypeError("max_hops must be an integer")

        if max_hops < 0:
            raise ValueError("max_hops must not be negative")

        if not isinstance(document_node_type, str):
            raise TypeError("document_node_type must be a string")

        if not document_node_type.strip():
            raise ValueError(
                "document_node_type must not be empty"
            )

        self._graph = graph
        self._traversal = GraphTraversal(graph)
        self._max_hops = max_hops
        self._document_node_type = document_node_type

        if start_nodes is None:
            self._start_nodes = tuple()
        else:
            self._start_nodes = tuple(start_nodes)

            for node_id in self._start_nodes:
                if not isinstance(node_id, str):
                    raise TypeError(
                        "start_nodes must contain strings"
                    )

                if not node_id.strip():
                    raise ValueError(
                        "start_nodes must not contain empty IDs"
                    )

                self._graph.require_node(node_id)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> Sequence[RetrievalResult]:
        """
        Retrieve documents reachable from the configured start nodes.

        The query is currently validated by the base retrieval contract,
        but does not directly influence traversal.

        This is intentional: query understanding and graph traversal
        are separate responsibilities.
        """

        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if not query.strip():
            raise ValueError("query must not be empty")

        if not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")

        if isinstance(top_k, bool):
            raise TypeError("top_k must be an integer")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        candidates: dict[str, float] = {}

        for start_node in self._start_nodes:
            for node_id, depth in self._traverse_with_depth(
                start_node
            ):
                node = self._graph.require_node(node_id)

                if node.node_type != self._document_node_type:
                    continue

                score = self._score(depth)

                previous_score = candidates.get(node_id)

                if (
                    previous_score is None
                    or score > previous_score
                ):
                    candidates[node_id] = score

        ranked = sorted(
            candidates.items(),
            key=lambda item: (-item[1], item[0]),
        )

        results: list[RetrievalResult] = []

        for document_id, score in ranked[:top_k]:
            node = self._graph.require_node(document_id)

            results.append(
                RetrievalResult(
                    document_id=document_id,
                    score=score,
                    metadata=node.metadata,
                )
            )

        return results

    def _traverse_with_depth(
        self,
        start_node: str,
    ) -> list[tuple[str, int]]:
        """
        Traverse from a starting node while preserving depth.
        """

        results: list[tuple[str, int]] = []

        for node_id in self._traversal.bfs(
            start_node,
            max_depth=self._max_hops,
        ):
            depth = self._distance_from_start(
                start_node,
                node_id,
            )

            results.append((node_id, depth))

        return results

    def _distance_from_start(
        self,
        start_node: str,
        target_node: str,
    ) -> int:
        """
        Calculate shortest hop distance between two reachable nodes.

        This helper keeps the implementation independent from the
        internal representation of GraphTraversal.
        """

        if start_node == target_node:
            return 0

        queue: list[tuple[str, int]] = [
            (start_node, 0)
        ]

        visited: set[str] = {start_node}

        while queue:
            current, depth = queue.pop(0)

            for neighbor in self._graph.neighbors(current):
                if neighbor in visited:
                    continue

                next_depth = depth + 1

                if neighbor == target_node:
                    return next_depth

                visited.add(neighbor)
                queue.append(
                    (neighbor, next_depth)
                )

        raise ValueError(
            f"Node {target_node!r} is not reachable "
            f"from {start_node!r}"
        )

    @staticmethod
    def _score(depth: int) -> float:
        """
        Convert graph distance into a relevance score.

        Closer documents receive higher scores.

            depth 0 -> 1.0
            depth 1 -> 0.5
            depth 2 -> 0.333...
            depth 3 -> 0.25
        """

        return 1.0 / (depth + 1)



    