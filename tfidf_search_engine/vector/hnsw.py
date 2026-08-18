from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from vector.similarity import VectorSimilarity


Number = int | float
Vector = Sequence[Number]


@dataclass
class HNSWNode:
    """
    Represents one node in the HNSW graph.

    A node stores only graph information.

    The actual vector remains owned by VectorStore.
    """

    id: str
    level: int
    neighbors: list[set[str]]

    @classmethod
    def create(
        cls,
        vector_id: str,
        level: int,
    ) -> HNSWNode:
        return cls(
            id=vector_id,
            level=level,
            neighbors=[set() for _ in range(level + 1)],
        )

    def neighbors_at(self, level: int) -> set[str]:
        if level < 0 or level > self.level:
            raise ValueError(
                f"invalid level {level} for node {self.id!r}"
            )

        return self.neighbors[level]


@dataclass(frozen=True)
class HNSWSearchResult:
    """
    Result returned by HNSW search.
    """

    id: str
    score: float


class HNSW:
    """
    Hierarchical Navigable Small World graph.

    HNSW is an approximate nearest-neighbor index.

    VectorStore remains the source of truth for:

        ID
        vector
        metadata

    HNSW owns:

        graph
        layers
        entry point
        ANN search parameters
    """

    def __init__(
        self,
        vector_store,
        *,
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        metric: str = "cosine",
        seed: int | None = None,
    ) -> None:

        self._validate_parameters(
            m=m,
            ef_construction=ef_construction,
            ef_search=ef_search,
            metric=metric,
        )

        self._vector_store = vector_store

        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.metric = metric

        self._nodes: dict[str, HNSWNode] = {}

        self._entry_point: str | None = None
        self._max_level = -1

        self._dimension: int | None = None

        self._random = random.Random(seed)

        # Probability distribution for node levels.
        #
        # level = floor(
        #     -log(U) / log(M)
        # )
        #
        # This makes higher levels exponentially rarer.
        self._level_multiplier = 1.0 / math.log(m)

    # ================================================================
    # PUBLIC API
    # ================================================================

    def add(
        self,
        vector_id: str,
        vector: Vector | None = None,
    ) -> None:
        """
        Add a vector to the HNSW graph.

        `vector` is optional because VectorStore owns the vector.

        It is accepted here so insertion can avoid an unnecessary second
        lookup when the caller already has the vector.
        """

        self._validate_id(vector_id)

        if vector_id in self._nodes:
            raise ValueError(
                f"vector ID already exists in HNSW: {vector_id!r}"
            )

        if vector is None:
            vector = self._get_vector(vector_id)

        self._validate_vector(vector)

        self._check_dimension(vector)

        level = self._random_level()

        node = HNSWNode.create(
            vector_id=vector_id,
            level=level,
        )

        self._nodes[vector_id] = node

        # First node.
        if self._entry_point is None:
            self._entry_point = vector_id
            self._max_level = level
            return

        current = self._entry_point

        # ------------------------------------------------------------
        # PHASE 1
        #
        # Start at the highest layer and greedily move toward the
        # new vector.
        #
        # We only do this for layers above the new node's level.
        # ------------------------------------------------------------

        for level_index in range(
            self._max_level,
            level,
            -1,
        ):
            current = self._greedy_search(
                query=vector,
                entry_point=current,
                level=level_index,
            )

        # ------------------------------------------------------------
        # PHASE 2
        #
        # Insert the node into every layer it belongs to.
        #
        # At these layers we perform a wider search using
        # ef_construction.
        # ------------------------------------------------------------

        upper_level = min(
            level,
            self._max_level,
        )

        for level_index in range(
            upper_level,
            -1,
            -1,
        ):
            candidates = self._search_layer(
                query=vector,
                entry_point=current,
                ef=self.ef_construction,
                level=level_index,
            )

            neighbors = self._select_neighbors(
                query=vector,
                candidates=candidates,
                max_neighbors=self._max_neighbors(level_index),
            )

            self._connect(
                vector_id=vector_id,
                neighbors=neighbors,
                level=level_index,
            )

            if candidates:
                current = self._nearest_id(
                    query=vector,
                    ids=candidates,
                )

        # ------------------------------------------------------------
        # If the new node reaches a higher layer than the current
        # entry point, it becomes the new entry point.
        # ------------------------------------------------------------

        if level > self._max_level:
            self._entry_point = vector_id
            self._max_level = level

    def add_batch(
        self,
        items: Sequence[tuple[str, Vector]],
    ) -> None:
        """
        Add multiple vectors.
        """

        for vector_id, vector in items:
            self.add(
                vector_id,
                vector,
            )

    def search(
        self,
        query: Vector,
        k: int = 10,
        *,
        ef_search: int | None = None,
    ) -> list[HNSWSearchResult]:
        """
        Perform approximate nearest-neighbor search.
        """

        self._validate_vector(query)

        if not isinstance(k, int) or isinstance(k, bool):
            raise TypeError("k must be an integer")

        if k < 1:
            raise ValueError("k must be >= 1")

        if not self._nodes:
            return []

        self._check_dimension(query)

        if ef_search is None:
            ef_search = self.ef_search

        if not isinstance(ef_search, int) or isinstance(
            ef_search,
            bool,
        ):
            raise TypeError("ef_search must be an integer")

        if ef_search < k:
            ef_search = k

        assert self._entry_point is not None

        current = self._entry_point

        # ------------------------------------------------------------
        # UPPER LAYERS
        #
        # Greedy search.
        # We only need one good entry point for the next layer.
        # ------------------------------------------------------------

        for level_index in range(
            self._max_level,
            0,
            -1,
        ):
            current = self._greedy_search(
                query=query,
                entry_point=current,
                level=level_index,
            )

        # ------------------------------------------------------------
        # LAYER 0
        #
        # Wider best-first search.
        # ------------------------------------------------------------

        candidates = self._search_layer(
            query=query,
            entry_point=current,
            ef=ef_search,
            level=0,
        )

        candidates.sort(
            key=lambda vector_id: self._ranking_key(
                self._score(
                    query,
                    self._get_vector(vector_id),
                )
            )
        )

        results: list[HNSWSearchResult] = []

        for vector_id in candidates[:k]:

            score = self._score(
                query,
                self._get_vector(vector_id),
            )

            results.append(
                HNSWSearchResult(
                    id=vector_id,
                    score=score,
                )
            )

        return results

    def remove(
        self,
        vector_id: str,
    ) -> None:
        """
        Remove a node from the graph.

        This removes all incoming and outgoing edges.

        Graph repair is intentionally not performed yet.
        """

        self._validate_id(vector_id)

        if vector_id not in self._nodes:
            raise KeyError(
                f"vector ID not found: {vector_id!r}"
            )

        node = self._nodes[vector_id]

        # Remove the node from every neighbor's adjacency list.
        for level in range(node.level + 1):

            for neighbor_id in list(
                node.neighbors[level]
            ):
                neighbor = self._nodes.get(neighbor_id)

                if neighbor is not None:
                    neighbor.neighbors[level].discard(
                        vector_id
                    )

        del self._nodes[vector_id]

        # Graph became empty.
        if not self._nodes:
            self._entry_point = None
            self._max_level = -1
            self._dimension = None
            return

        # Rebuild the entry point if necessary.
        if vector_id == self._entry_point:
            self._recalculate_entry_point()

    def contains(
        self,
        vector_id: str,
    ) -> bool:
        return vector_id in self._nodes

    def clear(self) -> None:
        self._nodes.clear()

        self._entry_point = None
        self._max_level = -1
        self._dimension = None

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(
        self,
        vector_id: str,
    ) -> bool:
        return vector_id in self._nodes

    def __repr__(self) -> str:
        return (
            f"HNSW("
            f"size={len(self)}, "
            f"metric={self.metric!r}, "
            f"m={self.m}, "
            f"ef_construction={self.ef_construction}, "
            f"ef_search={self.ef_search}, "
            f"max_level={self._max_level}"
            f")"
        )

    # ================================================================
    # LEVEL GENERATION
    # ================================================================

    def _random_level(self) -> int:
        """
        Randomly determine how many layers a node participates in.

        Mathematical idea:

            L = floor(
                -log(U) / log(M)
            )

        where:

            U ~ Uniform(0, 1)

        Higher levels therefore become exponentially rarer.
        """

        u = self._random.random()

        while u <= 0.0:
            u = self._random.random()

        return int(
            -math.log(u) * self._level_multiplier
        )

    # ================================================================
    # GREEDY SEARCH
    # ================================================================

    def _greedy_search(
        self,
        query: Vector,
        entry_point: str,
        level: int,
    ) -> str:
        """
        Greedily move toward the query at one layer.

        Algorithm:

            current = entry_point

            while a neighbor is better:
                move to that neighbor

            return current

        This is primarily used on upper layers.
        """

        current = entry_point

        current_score = self._score(
            query,
            self._get_vector(current),
        )

        while True:

            best = current
            best_score = current_score

            neighbors = self._nodes[
                current
            ].neighbors_at(level)

            for neighbor_id in neighbors:

                neighbor_score = self._score(
                    query,
                    self._get_vector(neighbor_id),
                )

                if self._is_better(
                    neighbor_score,
                    best_score,
                ):
                    best = neighbor_id
                    best_score = neighbor_score

            if best == current:
                break

            current = best
            current_score = best_score

        return current

    # ================================================================
    # BEST-FIRST SEARCH
    # ================================================================

    def _search_layer(
        self,
        query: Vector,
        entry_point: str,
        ef: int,
        level: int,
    ) -> list[str]:
        """
        Search one layer using a candidate frontier.

        This is the important part of HNSW search.

        Unlike greedy search, we don't immediately throw away every
        alternative path.

        We maintain:

            candidates
                nodes still worth exploring

            results
                best nodes discovered so far
        """

        visited: set[str] = {
            entry_point
        }

        candidates: list[str] = [
            entry_point
        ]

        results: set[str] = {
            entry_point
        }

        while candidates:

            current = self._pop_best_candidate(
                query=query,
                candidates=candidates,
            )

            current_score = self._score(
                query,
                self._get_vector(current),
            )

            if len(results) >= ef:

                worst_id = self._worst_id(
                    query=query,
                    ids=results,
                )

                worst_score = self._score(
                    query,
                    self._get_vector(worst_id),
                )

                # If current isn't better than the worst result,
                # no useful expansion is possible through this branch.
                if not self._is_better(
                    current_score,
                    worst_score,
                ):
                    break

            neighbors = self._nodes[
                current
            ].neighbors_at(level)

            for neighbor_id in neighbors:

                if neighbor_id in visited:
                    continue

                visited.add(neighbor_id)

                neighbor_score = self._score(
                    query,
                    self._get_vector(neighbor_id),
                )

                if len(results) < ef:

                    results.add(neighbor_id)
                    candidates.append(neighbor_id)

                    continue

                worst_id = self._worst_id(
                    query=query,
                    ids=results,
                )

                worst_score = self._score(
                    query,
                    self._get_vector(worst_id),
                )

                if self._is_better(
                    neighbor_score,
                    worst_score,
                ):
                    results.remove(worst_id)
                    results.add(neighbor_id)
                    candidates.append(neighbor_id)

        return list(results)

    # ================================================================
    # NEIGHBOR SELECTION
    # ================================================================

    def _select_neighbors(
        self,
        query: Vector,
        candidates: Sequence[str],
        max_neighbors: int,
    ) -> list[str]:
        """
        Select up to M neighbors for a node.

        First choose candidates by relevance.

        Then apply a diversity heuristic.

        The goal is not merely:

            "pick the M closest nodes"

        but:

            "pick useful and sufficiently diverse neighbors"
        """

        if not candidates:
            return []

        ordered = sorted(
            set(candidates),
            key=lambda vector_id: self._ranking_key(
                self._score(
                    query,
                    self._get_vector(vector_id),
                )
            ),
        )

        selected: list[str] = []

        for candidate_id in ordered:

            if len(selected) >= max_neighbors:
                break

            candidate_vector = self._get_vector(
                candidate_id
            )

            accept = True

            candidate_to_query = self._score(
                query,
                candidate_vector,
            )

            for selected_id in selected:

                selected_vector = self._get_vector(
                    selected_id
                )

                candidate_to_selected = self._score(
                    candidate_vector,
                    selected_vector,
                )

                # Diversity heuristic.
                #
                # If candidate is closer/more similar to an already
                # selected neighbor than it is to the query, the edge
                # may be redundant.
                if self._is_redundant(
                    candidate_to_query,
                    candidate_to_selected,
                ):
                    accept = False
                    break

            if accept:
                selected.append(candidate_id)

        # The heuristic should never prevent us from filling the
        # available degree unnecessarily.
        if len(selected) < max_neighbors:

            selected_set = set(selected)

            for candidate_id in ordered:

                if candidate_id in selected_set:
                    continue

                selected.append(candidate_id)

                if len(selected) >= max_neighbors:
                    break

        return selected[:max_neighbors]

    def _is_redundant(
        self,
        candidate_to_query: float,
        candidate_to_selected: float,
    ) -> bool:

        if self.metric in {"cosine", "dot"}:
            return candidate_to_selected > candidate_to_query

        return candidate_to_selected < candidate_to_query

    # ================================================================
    # GRAPH CONNECTION
    # ================================================================

    def _connect(
        self,
        vector_id: str,
        neighbors: Sequence[str],
        level: int,
    ) -> None:
        """
        Connect the new node bidirectionally.
        """

        node = self._nodes[vector_id]

        for neighbor_id in neighbors:

            if neighbor_id == vector_id:
                continue

            node.neighbors[level].add(
                neighbor_id
            )

            neighbor = self._nodes[neighbor_id]

            neighbor.neighbors[level].add(
                vector_id
            )

            max_neighbors = self._max_neighbors(
                level
            )

            # The neighbor may now have too many edges.
            if len(
                neighbor.neighbors[level]
            ) > max_neighbors:

                selected = self._select_neighbors(
                    query=self._get_vector(
                        neighbor_id
                    ),
                    candidates=list(
                        neighbor.neighbors[level]
                    ),
                    max_neighbors=max_neighbors,
                )

                neighbor.neighbors[level] = set(
                    selected
                )

    # ================================================================
    # GRAPH UTILITIES
    # ================================================================

    def _max_neighbors(
        self,
        level: int,
    ) -> int:
        """
        Layer 0 gets more connections.

        Upper layers:
            M

        Layer 0:
            2M
        """

        if level == 0:
            return 2 * self.m

        return self.m

    def _pop_best_candidate(
        self,
        query: Vector,
        candidates: list[str],
    ) -> str:
        """
        Remove and return the currently best candidate.
        """

        best_index = min(
            range(len(candidates)),
            key=lambda index: self._ranking_key(
                self._score(
                    query,
                    self._get_vector(
                        candidates[index]
                    ),
                )
            ),
        )

        return candidates.pop(best_index)

    def _worst_id(
        self,
        query: Vector,
        ids: Sequence[str] | set[str],
    ) -> str:
        """
        Return the least useful result.
        """

        return max(
            ids,
            key=lambda vector_id: self._ranking_key(
                self._score(
                    query,
                    self._get_vector(vector_id),
                )
            ),
        )

    def _nearest_id(
        self,
        query: Vector,
        ids: Sequence[str],
    ) -> str:
        return min(
            ids,
            key=lambda vector_id: self._ranking_key(
                self._score(
                    query,
                    self._get_vector(vector_id),
                )
            ),
        )

    # ================================================================
    # SIMILARITY
    # ================================================================

    def _score(
        self,
        vector_a: Vector,
        vector_b: Vector,
    ) -> float:
        """
        Delegate similarity mathematics to VectorSimilarity.
        """

        if self.metric == "cosine":
            return VectorSimilarity.cosine_similarity(
                vector_a,
                vector_b,
            )

        if self.metric == "dot":
            return VectorSimilarity.dot_product(
                vector_a,
                vector_b,
            )

        return VectorSimilarity.euclidean_distance(
            vector_a,
            vector_b,
        )

    def _is_better(
        self,
        score_a: float,
        score_b: float,
    ) -> bool:

        if self.metric in {"cosine", "dot"}:
            return score_a > score_b

        return score_a < score_b

    def _ranking_key(
        self,
        score: float,
    ) -> float:
        """
        Convert all metrics to:

            smaller = better
        """

        if self.metric in {"cosine", "dot"}:
            return -score

        return score

    # ================================================================
    # VECTORSTORE ACCESS
    # ================================================================

    def _get_vector(
        self,
        vector_id: str,
    ) -> Vector:
        return self._vector_store.get_vector(
            vector_id
        )

    # ================================================================
    # ENTRY POINT
    # ================================================================

    def _recalculate_entry_point(self) -> None:
        """
        Choose a remaining node with the highest level.
        """

        self._entry_point = max(
            self._nodes,
            key=lambda vector_id: self._nodes[
                vector_id
            ].level,
        )

        self._max_level = self._nodes[
            self._entry_point
        ].level

    # ================================================================
    # VALIDATION
    # ================================================================

    @staticmethod
    def _validate_parameters(
        *,
        m: int,
        ef_construction: int,
        ef_search: int,
        metric: str,
    ) -> None:

        if not isinstance(m, int) or isinstance(m, bool):
            raise TypeError("m must be an integer")

        if m < 2:
            raise ValueError("m must be >= 2")

        if not isinstance(
            ef_construction,
            int,
        ) or isinstance(
            ef_construction,
            bool,
        ):
            raise TypeError(
                "ef_construction must be an integer"
            )

        if ef_construction < m:
            raise ValueError(
                "ef_construction must be >= m"
            )

        if not isinstance(
            ef_search,
            int,
        ) or isinstance(
            ef_search,
            bool,
        ):
            raise TypeError(
                "ef_search must be an integer"
            )

        if ef_search < 1:
            raise ValueError(
                "ef_search must be >= 1"
            )

        if metric not in {
            "cosine",
            "dot",
            "euclidean",
        }:
            raise ValueError(
                "metric must be one of: "
                "cosine, dot, euclidean"
            )

    @staticmethod
    def _validate_id(
        vector_id: str,
    ) -> None:

        if not isinstance(
            vector_id,
            str,
        ):
            raise TypeError(
                "vector_id must be a string"
            )

        if not vector_id:
            raise ValueError(
                "vector_id must not be empty"
            )

    @staticmethod
    def _validate_vector(
        vector: Vector,
    ) -> None:

        if isinstance(
            vector,
            (str, bytes),
        ):
            raise TypeError(
                "vector must be a numeric sequence"
            )

        try:
            values = list(vector)
        except TypeError as exc:
            raise TypeError(
                "vector must be an iterable"
            ) from exc

        if not values:
            raise ValueError(
                "vector must not be empty"
            )

        for value in values:

            if isinstance(value, bool):
                raise TypeError(
                    "vector values cannot be bool"
                )

            if not isinstance(
                value,
                (int, float),
            ):
                raise TypeError(
                    "vector values must be numeric"
                )

            if not math.isfinite(
                float(value)
            ):
                raise ValueError(
                    "vector values must be finite"
                )

    def _check_dimension(
        self,
        vector: Vector,
    ) -> None:

        dimension = len(vector)

        if self._dimension is None:
            self._dimension = dimension
            return

        if dimension != self._dimension:
            raise ValueError(
                f"vector dimension mismatch: "
                f"expected {self._dimension}, "
                f"got {dimension}"
            )

    

