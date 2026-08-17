# Start at an entry point
#         ↓
# Look at its neighbors
#         ↓
# Move toward a neighbor closer to query
#         ↓
# Repeat
#         ↓
# Reach promising region
#         ↓
# Explore several candidates
#         ↓
# Return Top-K


# High HNSW layer
#        ↓
# Large jumps
#        ↓
# Lower layer
#        ↓
# More precise navigation
#        ↓
# Layer 0
#        ↓
# Local neighborhood
#        ↓
# Nearest vectors

# data structure as : vector id ---> vector's connections --> connection at each layer 

# node = {
#     "id": "doc42",
#     "levels": {
#         0: {"doc1", "doc7", "doc19"},
#         1: {"doc3", "doc8"},
#         2: {"doc91"},
#     }
# }

# doc1 → layer 0
# doc2 → layer 0
# doc3 → layers 0,1
# doc4 → layer 0
# doc5 → layers 0,1,2
# doc6 → layer 0

# Generate random level for X
#         ↓
# Find entry point
#         ↓
# Start at highest layer
#         ↓
# Greedily move toward X
#         ↓
# Drop down a layer
#         ↓
# Continue searching
#         ↓
# Reach layer 0
#         ↓
# Collect candidate neighbors
#         ↓
# Select appropriate neighbors
#         ↓
# Create bidirectional connections

from __future__ import annotations

import heapq
import math
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass


Number = int | float
Vector = Sequence[Number]
VectorGetter = Callable[[str], Vector]


@dataclass(frozen=True)
class HNSWSearchResult:
    """
    Result returned by an HNSW approximate nearest-neighbor search.

    Attributes:
        id:
            ID of the vector.
        score:
            Similarity score or distance, depending on the metric.
        """
    id: str
    score: float


class HNSW:
    """
    Hierarchical Navigable Small World graph.

    HNSW is an approximate nearest-neighbor (ANN) index over vectors.

    Important architectural rule:

        HNSW does NOT own the vector data.

    The caller remains responsible for storing vectors. HNSW only stores:

        vector IDs
        graph connectivity
        hierarchy
        entry point

    A vector_getter callback is used whenever HNSW needs the actual vector.

    Supported metrics:

        cosine
        dot
        euclidean

    For cosine and dot:

        higher score = more similar

    For euclidean:

        lower distance = more similar
    """

    def __init__(
        self,
        vector_getter: VectorGetter,
        *,
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        metric: str = "cosine",
        seed: int | None = None,
    ) -> None:

        if not callable(vector_getter):
            raise TypeError("vector_getter must be callable")

        if not isinstance(m, int) or isinstance(m, bool):
            raise TypeError("m must be an integer")

        if m < 2:
            raise ValueError("m must be >= 2")

        if not isinstance(ef_construction, int) or isinstance(
            ef_construction, bool
        ):
            raise TypeError("ef_construction must be an integer")

        if ef_construction < m:
            raise ValueError("ef_construction must be >= m")

        if not isinstance(ef_search, int) or isinstance(ef_search, bool):
            raise TypeError("ef_search must be an integer")

        if ef_search < 1:
            raise ValueError("ef_search must be >= 1")

        if metric not in {"cosine", "dot", "euclidean"}:
            raise ValueError(
                "metric must be one of: cosine, dot, euclidean"
            )

        self._vector_getter = vector_getter

        self.m = m
        self.m_max = m
        self.m_max_0 = 2 * m

        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.metric = metric

        self._nodes: dict[str, list[set[str]]] = {}

        self._entry_point: str | None = None
        self._max_level = -1

        self._dimension: int | None = None

        self._rng = random.Random(seed)

        # HNSW level generation:
        #
        #   P(level >= l) ~= exp(-l / mL)
        #
        # Standard HNSW implementations use:
        #
        #   mL = 1 / log(M)
        #
        # and:
        #
        #   level = floor(-log(U) * mL)
        #
        self._level_multiplier = 1.0 / math.log(self.m)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, vector_id: str, vector: Vector | None = None) -> None:
        """
        Add a vector ID to the HNSW graph.

        The vector itself remains owned by the external VectorStore.

        `vector` is accepted for insertion-time validation. If omitted,
        vector_getter(vector_id) is used.

        Duplicate IDs are rejected.
        """

        self._validate_id(vector_id)

        if vector_id in self._nodes:
            raise ValueError(f"vector ID already exists: {vector_id!r}")

        if vector is None:
            vector = self._get_vector(vector_id)

        self._validate_vector(vector)

        if self._dimension is None:
            self._dimension = len(vector)
        elif len(vector) != self._dimension:
            raise ValueError(
                f"vector dimension mismatch: expected "
                f"{self._dimension}, got {len(vector)}"
            )

        level = self._random_level()

        # Every node has a neighbor set for every layer it participates in.
        self._nodes[vector_id] = [
            set() for _ in range(level + 1)
        ]

        # First node becomes the entry point.
        if self._entry_point is None:
            self._entry_point = vector_id
            self._max_level = level
            return

        entry_point = self._entry_point

        # --------------------------------------------------------------
        # Phase 1:
        #
        # Starting from the current entry point, greedily descend through
        # layers above the new node's highest layer.
        # --------------------------------------------------------------

        if level < self._max_level:
            for layer in range(self._max_level, level, -1):
                entry_point = self._search_layer_greedy(
                    query=vector,
                    entry_points=[entry_point],
                    layer=layer,
                )

        # --------------------------------------------------------------
        # Phase 2:
        #
        # At every layer in which the new node exists:
        #
        #   1. perform a broader search
        #   2. choose neighbors
        #   3. connect the new node
        #   4. connect those neighbors back to the new node
        # --------------------------------------------------------------

        upper_layer = min(level, self._max_level)

        for layer in range(upper_layer, -1, -1):

            candidates = self._search_layer(
                query=vector,
                entry_points=[entry_point],
                ef=self.ef_construction,
                layer=layer,
            )

            selected = self._select_neighbors(
                query=vector,
                candidates=candidates,
                max_neighbors=self._max_neighbors(layer),
            )

            self._connect_new_node(
                vector_id=vector_id,
                neighbors=selected,
                layer=layer,
            )

            # The best candidate from this layer becomes the entry point
            # for the next lower layer.
            if candidates:
                entry_point = self._closest_candidate(
                    query=vector,
                    candidates=candidates,
                )

        # --------------------------------------------------------------
        # If the new node is higher than the previous entry point,
        # it becomes the new global entry point.
        # --------------------------------------------------------------

        if level > self._max_level:
            self._entry_point = vector_id
            self._max_level = level

    def add_batch(
        self,
        items: Iterable[tuple[str, Vector]],
    ) -> None:
        """
        Add multiple vectors.

        Each item must be:

            (vector_id, vector)
        """

        for vector_id, vector in items:
            self.add(vector_id, vector)

    def search(
        self,
        query: Vector,
        k: int = 10,
        *,
        ef_search: int | None = None,
    ) -> list[HNSWSearchResult]:
        """
        Approximate nearest-neighbor search.

        Returns up to `k` results.

        `ef_search` may override the configured search breadth for one query.
        """

        self._validate_vector(query)

        if not isinstance(k, int) or isinstance(k, bool):
            raise TypeError("k must be an integer")

        if k < 1:
            raise ValueError("k must be >= 1")

        if self._dimension is not None and len(query) != self._dimension:
            raise ValueError(
                f"query dimension mismatch: expected "
                f"{self._dimension}, got {len(query)}"
            )

        if self._entry_point is None:
            return []

        if ef_search is None:
            ef_search = self.ef_search

        if not isinstance(ef_search, int) or isinstance(ef_search, bool):
            raise TypeError("ef_search must be an integer")

        if ef_search < k:
            ef_search = k

        # --------------------------------------------------------------
        # Phase 1:
        #
        # Greedy descent through upper layers.
        # --------------------------------------------------------------

        entry_point = self._entry_point

        for layer in range(self._max_level, 0, -1):
            entry_point = self._search_layer_greedy(
                query=query,
                entry_points=[entry_point],
                layer=layer,
            )

        # --------------------------------------------------------------
        # Phase 2:
        #
        # Wider exploration at layer 0.
        # --------------------------------------------------------------

        candidates = self._search_layer(
            query=query,
            entry_points=[entry_point],
            ef=ef_search,
            layer=0,
        )

        ranked = sorted(
            candidates,
            key=lambda node_id: self._ranking_key(
                self._score(query, self._get_vector(node_id))
            ),
        )

        results = [
            HNSWSearchResult(
                id=node_id,
                score=self._score(query, self._get_vector(node_id)),
            )
            for node_id in ranked[:k]
        ]

        return results

    def remove(self, vector_id: str) -> None:
        """
        Remove a node from the graph.

        HNSW deletion is implemented by removing the node from every
        neighbor's adjacency set.

        This implementation does not attempt graph repair/reinsertion.
        """

        self._validate_id(vector_id)

        if vector_id not in self._nodes:
            raise KeyError(f"unknown vector ID: {vector_id!r}")

        levels = self._nodes[vector_id]

        for layer, neighbors in enumerate(levels):

            for neighbor_id in list(neighbors):

                if neighbor_id in self._nodes:
                    self._nodes[neighbor_id][layer].discard(vector_id)

        del self._nodes[vector_id]

        # Empty graph.
        if not self._nodes:
            self._entry_point = None
            self._max_level = -1
            self._dimension = None
            return

        # If the entry point was deleted, choose the highest-level
        # remaining node.
        if vector_id == self._entry_point:

            new_entry_point = max(
                self._nodes,
                key=lambda node_id: len(self._nodes[node_id]),
            )

            self._entry_point = new_entry_point
            self._max_level = len(
                self._nodes[new_entry_point]
            ) - 1

    def contains(self, vector_id: str) -> bool:
        """Return whether a vector ID exists in the graph."""

        self._validate_id(vector_id)
        return vector_id in self._nodes

    def clear(self) -> None:
        """Remove every node from the graph."""

        self._nodes.clear()
        self._entry_point = None
        self._max_level = -1
        self._dimension = None

    def __len__(self) -> int:
        """Return the number of indexed vectors."""

        return len(self._nodes)

    def __contains__(self, vector_id: str) -> bool:
        return vector_id in self._nodes

    def __repr__(self) -> str:
        return (
            f"HNSW("
            f"size={len(self)}, "
            f"dimension={self._dimension}, "
            f"metric={self.metric!r}, "
            f"m={self.m}, "
            f"ef_construction={self.ef_construction}, "
            f"ef_search={self.ef_search}, "
            f"max_level={self._max_level}"
            f")"
        )

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _random_level(self) -> int:
        """
        Generate the maximum layer for a new node.

        Higher levels become exponentially rarer.
        """

        random_value = self._rng.random()

        # random() can theoretically return 0.
        while random_value <= 0.0:
            random_value = self._rng.random()

        return int(
            -math.log(random_value) * self._level_multiplier
        )

    def _connect_new_node(
        self,
        vector_id: str,
        neighbors: Sequence[str],
        layer: int,
    ) -> None:
        """
        Create bidirectional edges between the new node and its neighbors.
        """

        node_neighbors = self._nodes[vector_id][layer]

        for neighbor_id in neighbors:

            if neighbor_id == vector_id:
                continue

            node_neighbors.add(neighbor_id)

            neighbor_neighbors = self._nodes[neighbor_id][layer]
            neighbor_neighbors.add(vector_id)

            max_neighbors = self._max_neighbors(layer)

            if len(neighbor_neighbors) > max_neighbors:

                neighbor_vector = self._get_vector(neighbor_id)

                candidates = [
                    (
                        candidate_id,
                        self._score(
                            neighbor_vector,
                            self._get_vector(candidate_id),
                        ),
                    )
                    for candidate_id in neighbor_neighbors
                    if candidate_id != neighbor_id
                ]

                selected = self._select_neighbors(
                    query=neighbor_vector,
                    candidates=candidates,
                    max_neighbors=max_neighbors,
                )

                neighbor_neighbors.clear()
                neighbor_neighbors.update(selected)

    def _select_neighbors(
        self,
        query: Vector,
        candidates: Iterable[str | tuple[str, float]],
        max_neighbors: int,
    ) -> list[str]:
        """
        Select neighbors for a node.

        This uses a diversity-aware heuristic rather than simply selecting
        the globally closest M nodes.

        Candidate A is accepted if it is sufficiently useful relative to
        already-selected neighbors.

        This is an educational implementation of the HNSW neighbor
        selection idea rather than a byte-for-byte reproduction of a
        particular production library.
        """

        normalized: list[tuple[str, float]] = []

        for candidate in candidates:

            if isinstance(candidate, tuple):
                candidate_id, score = candidate
            else:
                candidate_id = candidate
                score = self._score(
                    query,
                    self._get_vector(candidate_id),
                )

            normalized.append((candidate_id, score))

        if not normalized:
            return []

        normalized.sort(
            key=lambda item: self._ranking_key(item[1])
        )

        selected: list[str] = []

        for candidate_id, candidate_score in normalized:

            if candidate_id not in self._nodes:
                continue

            if len(selected) >= max_neighbors:
                break

            accept = True

            for selected_id in selected:

                selected_score = self._score(
                    self._get_vector(candidate_id),
                    self._get_vector(selected_id),
                )

                # For similarity metrics, a candidate that is very similar
                # to an already-selected neighbor may provide little graph
                # diversity.
                #
                # For distance metrics, the same idea is expressed with
                # smaller distances.
                if self._candidate_is_redundant(
                    candidate_score=candidate_score,
                    candidate_to_selected=selected_score,
                ):
                    accept = False
                    break

            if accept:
                selected.append(candidate_id)

        # If the diversity heuristic was too aggressive, fill remaining
        # capacity with closest candidates.
        if len(selected) < max_neighbors:

            selected_set = set(selected)

            for candidate_id, _ in normalized:

                if candidate_id in selected_set:
                    continue

                selected.append(candidate_id)

                if len(selected) >= max_neighbors:
                    break

        return selected[:max_neighbors]

    def _candidate_is_redundant(
        self,
        *,
        candidate_score: float,
        candidate_to_selected: float,
    ) -> bool:
        """
        Determine whether a candidate is redundant.

        Similarity:
            candidate-to-selected > candidate-to-query

        Distance:
            candidate-to-selected < candidate-to-query
        """

        if self.metric in {"cosine", "dot"}:
            return candidate_to_selected > candidate_score

        return candidate_to_selected < candidate_score

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _search_layer_greedy(
        self,
        query: Vector,
        entry_points: Sequence[str],
        layer: int,
    ) -> str:
        """
        Greedy search used for upper HNSW layers.

        At these layers we primarily want to find a good starting point
        for the next layer.
        """

        current = entry_points[0]

        current_score = self._score(
            query,
            self._get_vector(current),
        )

        improved = True

        while improved:

            improved = False

            for neighbor_id in self._nodes[current][layer]:

                neighbor_score = self._score(
                    query,
                    self._get_vector(neighbor_id),
                )

                if self._is_better(
                    neighbor_score,
                    current_score,
                ):
                    current = neighbor_id
                    current_score = neighbor_score
                    improved = True
                    break

        return current

    def _search_layer(
        self,
        query: Vector,
        entry_points: Sequence[str],
        ef: int,
        layer: int,
    ) -> list[str]:
        """
        Best-first graph search.

        Maintains two structures:

            candidates:
                nodes still worth exploring

            results:
                best nodes discovered so far

        This is the key search mechanism that makes HNSW more robust than
        pure greedy descent.
        """

        if not entry_points:
            return []

        visited: set[str] = set()

        # Python's heapq is a min-heap.
        #
        # Candidate heap:
        #
        #   (distance-like key, node_id)
        #
        # We transform similarity scores into a value where smaller is
        # better for heap operations.
        candidates: list[tuple[float, str]] = []

        # Results heap:
        #
        # Keep the WORST result at the top so it can be removed efficiently.
        results: list[tuple[float, str]] = []

        for entry_point in entry_points:

            if entry_point not in self._nodes:
                continue

            if layer >= len(self._nodes[entry_point]):
                continue

            score = self._score(
                query,
                self._get_vector(entry_point),
            )

            heap_key = self._heap_key(score)

            heapq.heappush(
                candidates,
                (heap_key, entry_point),
            )

            # For results we need a key where the worst item is easiest
            # to identify.
            result_key = self._worst_heap_key(score)

            heapq.heappush(
                results,
                (result_key, entry_point),
            )

            visited.add(entry_point)

        while candidates:

            candidate_heap_key, current_id = heapq.heappop(
                candidates
            )

            current_score = self._score(
                query,
                self._get_vector(current_id),
            )

            if len(results) >= ef:

                worst_id = results[0][1]

                worst_score = self._score(
                    query,
                    self._get_vector(worst_id),
                )

                if not self._is_better(
                    current_score,
                    worst_score,
                ):
                    break

            for neighbor_id in self._nodes[current_id][layer]:

                if neighbor_id in visited:
                    continue

                visited.add(neighbor_id)

                neighbor_score = self._score(
                    query,
                    self._get_vector(neighbor_id),
                )

                if len(results) < ef:

                    heapq.heappush(
                        candidates,
                        (
                            self._heap_key(neighbor_score),
                            neighbor_id,
                        ),
                    )

                    heapq.heappush(
                        results,
                        (
                            self._worst_heap_key(neighbor_score),
                            neighbor_id,
                        ),
                    )

                else:

                    worst_id = results[0][1]

                    worst_score = self._score(
                        query,
                        self._get_vector(worst_id),
                    )

                    if self._is_better(
                        neighbor_score,
                        worst_score,
                    ):
                        heapq.heappush(
                            candidates,
                            (
                                self._heap_key(neighbor_score),
                                neighbor_id,
                            ),
                        )

                        heapq.heapreplace(
                            results,
                            (
                                self._worst_heap_key(neighbor_score),
                                neighbor_id,
                            ),
                        )

        return [node_id for _, node_id in results]

    # ------------------------------------------------------------------
    # Metric operations
    # ------------------------------------------------------------------

    def _score(
        self,
        vector_a: Vector,
        vector_b: Vector,
    ) -> float:
        """
        Calculate the configured metric.

        The sign/order is preserved:

            cosine -> larger is better
            dot    -> larger is better
            euclidean -> smaller is better
        """

        self._validate_vector(vector_a)
        self._validate_vector(vector_b)

        if len(vector_a) != len(vector_b):
            raise ValueError(
                "vectors must have the same dimension"
            )

        if self.metric == "dot":

            return float(
                sum(
                    float(a) * float(b)
                    for a, b in zip(
                        vector_a,
                        vector_b,
                        strict=True,
                    )
                )
            )

        if self.metric == "cosine":

            dot = sum(
                float(a) * float(b)
                for a, b in zip(
                    vector_a,
                    vector_b,
                    strict=True,
                )
            )

            norm_a = math.sqrt(
                sum(float(a) ** 2 for a in vector_a)
            )

            norm_b = math.sqrt(
                sum(float(b) ** 2 for b in vector_b)
            )

            if norm_a == 0.0 or norm_b == 0.0:
                raise ValueError(
                    "cosine similarity is undefined for zero vectors"
                )

            return dot / (norm_a * norm_b)

        return math.sqrt(
            sum(
                (float(a) - float(b)) ** 2
                for a, b in zip(
                    vector_a,
                    vector_b,
                    strict=True,
                )
            )
        )

    def _is_better(
        self,
        score_a: float,
        score_b: float,
    ) -> bool:
        if self.metric in {"cosine", "dot"}:
            return score_a > score_b

        return score_a < score_b

    def _ranking_key(self, score: float) -> float:
        """
        Smaller values are better for sorting.
        """

        if self.metric in {"cosine", "dot"}:
            return -score

        return score

    def _heap_key(self, score: float) -> float:
        """
        Candidate heap key.

        Smaller = better.
        """

        return self._ranking_key(score)

    def _worst_heap_key(self, score: float) -> float:
        """
        Results heap key.

        Larger = better because heap root should represent the worst
        result.

        Therefore:

            similarity:
                score itself

            distance:
                -distance
        """

        if self.metric in {"cosine", "dot"}:
            return score

        return -score

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _closest_candidate(
        self,
        query: Vector,
        candidates: Iterable[str],
    ) -> str:
        return min(
            candidates,
            key=lambda node_id: self._ranking_key(
                self._score(
                    query,
                    self._get_vector(node_id),
                )
            ),
        )

    def _max_neighbors(self, layer: int) -> int:
        """
        Layer 0 normally has a larger maximum degree.

            upper layers -> M
            layer 0      -> 2M
        """

        if layer == 0:
            return self.m_max_0

        return self.m_max

    def _get_vector(self, vector_id: str) -> Vector:
        vector = self._vector_getter(vector_id)

        self._validate_vector(vector)

        if self._dimension is not None and len(vector) != self._dimension:
            raise ValueError(
                f"vector dimension mismatch for {vector_id!r}: "
                f"expected {self._dimension}, got {len(vector)}"
            )

        return vector

    @staticmethod
    def _validate_id(vector_id: str) -> None:
        if not isinstance(vector_id, str):
            raise TypeError("vector_id must be a string")

        if not vector_id:
            raise ValueError("vector_id must not be empty")

    @staticmethod
    def _validate_vector(vector: Vector) -> None:
        if isinstance(vector, (str, bytes)):
            raise TypeError("vector must be a numeric sequence")

        try:
            values = list(vector)
        except TypeError as exc:
            raise TypeError(
                "vector must be an iterable of numbers"
            ) from exc

        if not values:
            raise ValueError("vector must not be empty")

        for value in values:

            if isinstance(value, bool):
                raise TypeError(
                    "vector values must be numeric; bool is not allowed"
                )

            if not isinstance(value, (int, float)):
                raise TypeError(
                    "vector values must be int or float"
                )

            if not math.isfinite(float(value)):
                raise ValueError(
                    "vector values must be finite"
                )




