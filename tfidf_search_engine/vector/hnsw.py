from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from vector.similarity import VectorSimilarity


# ================================================================
# TYPE ALIASES
# ================================================================

Number = int | float
Vector = Sequence[Number]


# ================================================================
# DATA STRUCTURES
# ================================================================

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
    ) -> "HNSWNode":

        return cls(
            id=vector_id,
            level=level,
            neighbors=[
                set()
                for _ in range(level + 1)
            ],
        )

    def neighbors_at(
        self,
        level: int,
    ) -> set[str]:

        if level < 0 or level > self.level:
            raise ValueError(
                f"invalid level {level} "
                f"for node {self.id!r}"
            )

        return self.neighbors[level]


@dataclass(frozen=True)
class HNSWSearchResult:
    """
    Result returned by HNSW search.
    """

    id: str
    score: float


# ================================================================
# HNSW
# ================================================================

class HNSW:
    """
    Hierarchical Navigable Small World graph.

    HNSW is an approximate nearest-neighbor index.

    VectorStore owns:

        - IDs
        - vectors
        - metadata

    HNSW owns:

        - graph structure
        - layers
        - entry point
        - ANN search parameters
    """

    # ============================================================
    # INITIALIZATION
    # ============================================================

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

        # Higher levels become exponentially rarer.
        self._level_multiplier = (
            1.0 / math.log(m)
        )

    # ============================================================
    # VALIDATION
    #
    # These are the lowest-level helpers.
    # ============================================================

    @staticmethod
    def _validate_parameters(
        *,
        m: int,
        ef_construction: int,
        ef_search: int,
        metric: str,
    ) -> None:

        if (
            not isinstance(m, int)
            or isinstance(m, bool)
        ):
            raise TypeError(
                "m must be an integer"
            )

        if m < 2:
            raise ValueError(
                "m must be >= 2"
            )

        if (
            not isinstance(
                ef_construction,
                int,
            )
            or isinstance(
                ef_construction,
                bool,
            )
        ):
            raise TypeError(
                "ef_construction must be an integer"
            )

        if ef_construction < m:
            raise ValueError(
                "ef_construction must be >= m"
            )

        if (
            not isinstance(
                ef_search,
                int,
            )
            or isinstance(
                ef_search,
                bool,
            )
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

    # ============================================================
    # VECTOR ACCESS
    #
    # Everything that needs a vector goes through this.
    # ============================================================

    def _get_vector(
        self,
        vector_id: str,
    ) -> Vector:

        return self._vector_store.get_vector(
            vector_id
        )

    # ============================================================
    # SIMILARITY AND RANKING
    #
    # These define what "better" means.
    # ============================================================

    def _score(
        self,
        vector_a: Vector,
        vector_b: Vector,
    ) -> float:

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

        if self.metric in {
            "cosine",
            "dot",
        }:
            return score_a > score_b

        return score_a < score_b

    def _ranking_key(
        self,
        score: float,
    ) -> float:
        """
        Convert every metric to:

            smaller = better
        """

        if self.metric in {
            "cosine",
            "dot",
        }:
            return -score

        return score

    # ============================================================
    # BASIC GRAPH UTILITIES
    #
    # Small reusable operations used by search and insertion.
    # ============================================================

    def _max_neighbors(
        self,
        level: int,
    ) -> int:
        """
        Layer 0 gets twice as many connections.
        """

        if level == 0:
            return 2 * self.m

        return self.m

    def _nearest_id(
        self,
        query: Vector,
        ids: Sequence[str],
    ) -> str:
        """
        Return the best node relative to the query.
        """

        return min(
            ids,
            key=lambda vector_id:
                self._ranking_key(
                    self._score(
                        query,
                        self._get_vector(
                            vector_id
                        ),
                    )
                ),
        )

    def _worst_id(
        self,
        query: Vector,
        ids: Sequence[str] | set[str],
    ) -> str:
        """
        Return the least useful node relative to the query.
        """

        return max(
            ids,
            key=lambda vector_id:
                self._ranking_key(
                    self._score(
                        query,
                        self._get_vector(
                            vector_id
                        ),
                    )
                ),
        )

    def _pop_best_candidate(
        self,
        query: Vector,
        candidates: list[str],
    ) -> str:
        """
        Remove and return the best unexplored candidate.
        """

        best_index = min(
            range(len(candidates)),
            key=lambda index:
                self._ranking_key(
                    self._score(
                        query,
                        self._get_vector(
                            candidates[index]
                        ),
                    )
                ),
        )

        return candidates.pop(
            best_index
        )

    # ============================================================
    # LEVEL GENERATION
    #
    # Determines which layers a new node belongs to.
    # ============================================================

    def _random_level(self) -> int:
        """
        Randomly determine the highest layer of a node.

        Higher levels become exponentially rarer.
        """

        u = self._random.random()

        while u <= 0.0:
            u = self._random.random()

        return int(
            -math.log(u)
            * self._level_multiplier
        )

    # ============================================================
    # SEARCH PRIMITIVE 1: GREEDY SEARCH
    #
    # Used on upper layers.
    # ============================================================

    def _greedy_search(
        self,
        query: Vector,
        entry_point: str,
        level: int,
    ) -> str:
        """
        Greedily move toward the query at one layer.

        Continue moving while a neighbor is better than
        the current node.
        """

        current = entry_point

        current_score = self._score(
            query,
            self._get_vector(
                current
            ),
        )

        while True:

            best = current
            best_score = current_score

            neighbors = self._nodes[
                current
            ].neighbors_at(
                level
            )

            for neighbor_id in neighbors:

                neighbor_score = self._score(
                    query,
                    self._get_vector(
                        neighbor_id
                    ),
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

    # ============================================================
    # SEARCH PRIMITIVE 2: BEST-FIRST SEARCH
    #
    # Used for ef-based exploration of a layer.
    # ============================================================

    def _search_layer(
        self,
        query: Vector,
        entry_point: str,
        ef: int,
        level: int,
    ) -> list[str]:
        """
        Search one HNSW layer.

        Maintains:

        candidates
            Nodes waiting to be explored.

        results
            The best `ef` nodes discovered so far.
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
                self._get_vector(
                    current
                ),
            )

            if len(results) >= ef:

                worst_id = self._worst_id(
                    query=query,
                    ids=results,
                )

                worst_score = self._score(
                    query,
                    self._get_vector(
                        worst_id
                    ),
                )

                if not self._is_better(
                    current_score,
                    worst_score,
                ):
                    break

            neighbors = self._nodes[
                current
            ].neighbors_at(
                level
            )

            for neighbor_id in neighbors:

                if neighbor_id in visited:
                    continue

                visited.add(
                    neighbor_id
                )

                neighbor_score = self._score(
                    query,
                    self._get_vector(
                        neighbor_id
                    ),
                )

                should_add = False

                if len(results) < ef:
                    should_add = True

                else:

                    worst_id = self._worst_id(
                        query=query,
                        ids=results,
                    )

                    worst_score = self._score(
                        query,
                        self._get_vector(
                            worst_id
                        ),
                    )

                    if self._is_better(
                        neighbor_score,
                        worst_score,
                    ):
                        results.remove(
                            worst_id
                        )

                        should_add = True

                if should_add:

                    results.add(
                        neighbor_id
                    )

                    candidates.append(
                        neighbor_id
                    )

        return list(results)

    # ============================================================
    # NEIGHBOR SELECTION
    #
    # Decides which discovered nodes should become graph neighbors.
    # ============================================================

    def _is_redundant(
        self,
        candidate_to_query: float,
        candidate_to_selected: float,
    ) -> bool:

        if self.metric in {
            "cosine",
            "dot",
        }:
            return (
                candidate_to_selected
                > candidate_to_query
            )

        return (
            candidate_to_selected
            < candidate_to_query
        )

    def _select_neighbors(
        self,
        query: Vector,
        candidates: Sequence[str],
        max_neighbors: int,
    ) -> list[str]:
        """
        Select useful and diverse neighbors.

        Candidates are first ordered by closeness
        to the query.
        """

        if not candidates:
            return []

        ordered = sorted(
            set(candidates),
            key=lambda vector_id:
                self._ranking_key(
                    self._score(
                        query,
                        self._get_vector(
                            vector_id
                        ),
                    )
                ),
        )

        selected: list[str] = []

        rejected: list[str] = []

        for candidate_id in ordered:

            if len(selected) >= max_neighbors:
                break

            candidate_vector = self._get_vector(
                candidate_id
            )

            candidate_to_query = self._score(
                query,
                candidate_vector,
            )

            accept = True

            for selected_id in selected:

                selected_vector = self._get_vector(
                    selected_id
                )

                candidate_to_selected = self._score(
                    candidate_vector,
                    selected_vector,
                )

                if self._is_redundant(
                    candidate_to_query,
                    candidate_to_selected,
                ):
                    accept = False
                    break

            if accept:
                selected.append(
                    candidate_id
                )

            else:
                rejected.append(
                    candidate_id
                )

        if len(selected) < max_neighbors:

            for candidate_id in rejected:

                if candidate_id in selected:
                    continue

                selected.append(
                    candidate_id
                )

                if (
                    len(selected)
                    >= max_neighbors
                ):
                    break

        return selected[:max_neighbors]

    # ============================================================
    # GRAPH CONNECTION AND PRUNING
    #
    # After neighbors are selected, actual graph edges are created.
    # ============================================================

    def _connect(
        self,
        vector_id: str,
        neighbors: Sequence[str],
        level: int,
    ) -> None:
        """
        Connect a node bidirectionally.
        """

        node = self._nodes[
            vector_id
        ]

        for neighbor_id in neighbors:

            if neighbor_id == vector_id:
                continue

            neighbor = self._nodes[
                neighbor_id
            ]

            if level > neighbor.level:
                continue

            node.neighbors_at(
                level
            ).add(
                neighbor_id
            )

            neighbor.neighbors_at(
                level
            ).add(
                vector_id
            )

            self._prune_neighbors(
                vector_id=neighbor_id,
                level=level,
            )

        self._prune_neighbors(
            vector_id=vector_id,
            level=level,
        )

    def _prune_neighbors(
        self,
        vector_id: str,
        level: int,
    ) -> None:
        """
        Reduce a node's degree if it exceeds
        the allowed maximum.
        """

        node = self._nodes[
            vector_id
        ]

        max_neighbors = self._max_neighbors(
            level
        )

        if (
            len(
                node.neighbors_at(
                    level
                )
            )
            <= max_neighbors
        ):
            return

        node_vector = self._get_vector(
            vector_id
        )

        current_neighbors = list(
            node.neighbors_at(
                level
            )
        )

        selected = self._select_neighbors(
            query=node_vector,
            candidates=current_neighbors,
            max_neighbors=max_neighbors,
        )

        old_neighbors = set(
            node.neighbors_at(
                level
            )
        )

        new_neighbors = set(
            selected
        )

        removed_neighbors = (
            old_neighbors
            - new_neighbors
        )

        node.neighbors[level] = (
            new_neighbors
        )

        for removed_id in removed_neighbors:

            removed_node = self._nodes.get(
                removed_id
            )

            if removed_node is not None:

                removed_node.neighbors_at(
                    level
                ).discard(
                    vector_id
                )

    # ============================================================
    # ENTRY POINT MANAGEMENT
    # ============================================================

    def _recalculate_entry_point(
        self,
    ) -> None:
        """
        Choose a remaining node with the highest level.
        """

        self._entry_point = max(
            self._nodes,
            key=lambda vector_id:
                self._nodes[
                    vector_id
                ].level,
        )

        self._max_level = (
            self._nodes[
                self._entry_point
            ].level
        )

    # ============================================================
    # PUBLIC API: INSERTION
    #
    # All insertion building blocks have now been defined above.
    # ============================================================

    def add(
        self,
        vector_id: str,
        vector: Vector | None = None,
    ) -> None:
        """
        Add a vector to the HNSW graph.

        `vector` is optional because VectorStore
        owns the vector.
        """

        self._validate_id(
            vector_id
        )

        if vector_id in self._nodes:
            raise ValueError(
                "vector ID already exists "
                f"in HNSW: {vector_id!r}"
            )

        if vector is None:
            vector = self._get_vector(
                vector_id
            )

        self._validate_vector(
            vector
        )

        self._check_dimension(
            vector
        )

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

        assert self._entry_point is not None

        current = self._entry_point

        # --------------------------------------------------------
        # PHASE 1
        #
        # Greedy traversal through layers above
        # the new node's highest layer.
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # PHASE 2
        #
        # Search and connect on every layer
        # where the new node exists.
        # --------------------------------------------------------

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
                max_neighbors=self._max_neighbors(
                    level_index
                ),
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

        # --------------------------------------------------------
        # New highest-level node becomes entry point.
        # --------------------------------------------------------

        if level > self._max_level:

            self._entry_point = vector_id
            self._max_level = level

    def add_batch(
        self,
        items: Sequence[
            tuple[str, Vector]
        ],
    ) -> None:
        """
        Add multiple vectors.
        """

        for vector_id, vector in items:

            self.add(
                vector_id,
                vector,
            )

    # ============================================================
    # PUBLIC API: SEARCH
    #
    # Search primitives were already introduced above.
    # ============================================================

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

        self._validate_vector(
            query
        )

        if (
            not isinstance(k, int)
            or isinstance(k, bool)
        ):
            raise TypeError(
                "k must be an integer"
            )

        if k < 1:
            raise ValueError(
                "k must be >= 1"
            )

        if not self._nodes:
            return []

        self._check_dimension(
            query
        )

        if ef_search is None:
            ef_search = self.ef_search

        if (
            not isinstance(
                ef_search,
                int,
            )
            or isinstance(
                ef_search,
                bool,
            )
        ):
            raise TypeError(
                "ef_search must be an integer"
            )

        if ef_search < 1:
            raise ValueError(
                "ef_search must be >= 1"
            )

        ef_search = max(
            ef_search,
            k,
        )

        assert self._entry_point is not None

        current = self._entry_point

        # --------------------------------------------------------
        # UPPER LAYERS
        #
        # Greedy traversal.
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # LAYER 0
        #
        # Wider best-first search.
        # --------------------------------------------------------

        candidates = self._search_layer(
            query=query,
            entry_point=current,
            ef=ef_search,
            level=0,
        )

        ordered = sorted(
            candidates,
            key=lambda vector_id:
                self._ranking_key(
                    self._score(
                        query,
                        self._get_vector(
                            vector_id
                        ),
                    )
                ),
        )

        results: list[
            HNSWSearchResult
        ] = []

        for vector_id in ordered[:k]:

            score = self._score(
                query,
                self._get_vector(
                    vector_id
                ),
            )

            results.append(
                HNSWSearchResult(
                    id=vector_id,
                    score=score,
                )
            )

        return results

    # ============================================================
    # PUBLIC API: REMOVAL
    # ============================================================

    def remove(
        self,
        vector_id: str,
    ) -> None:
        """
        Remove a node from the graph.

        All incoming and outgoing edges are removed.
        """

        self._validate_id(
            vector_id
        )

        if vector_id not in self._nodes:
            raise KeyError(
                f"vector ID not found: "
                f"{vector_id!r}"
            )

        node = self._nodes[
            vector_id
        ]

        # Remove all reverse edges.
        for level in range(
            node.level + 1
        ):

            for neighbor_id in list(
                node.neighbors[level]
            ):

                neighbor = self._nodes.get(
                    neighbor_id
                )

                if neighbor is not None:

                    neighbor.neighbors_at(
                        level
                    ).discard(
                        vector_id
                    )

        del self._nodes[
            vector_id
        ]

        # Graph became empty.
        if not self._nodes:

            self._entry_point = None
            self._max_level = -1
            self._dimension = None

            return

        # Recalculate entry point if necessary.
        if vector_id == self._entry_point:

            self._recalculate_entry_point()

    # ============================================================
    # PUBLIC API: BASIC OPERATIONS
    # ============================================================

    def contains(
        self,
        vector_id: str,
    ) -> bool:

        return vector_id in self._nodes

    def clear(
        self,
    ) -> None:

        self._nodes.clear()

        self._entry_point = None
        self._max_level = -1
        self._dimension = None

    def __len__(
        self,
    ) -> int:
        return len(
            self._nodes
        )

    def __contains__(
        self,
        vector_id: str,
    ) -> bool:

        return vector_id in self._nodes

    def __repr__(
        self,
    ) -> str:

        return (
            f"HNSW("
            f"size={len(self)}, "
            f"metric={self.metric!r}, "
            f"m={self.m}, "
            f"ef_construction="
            f"{self.ef_construction}, "
            f"ef_search={self.ef_search}, "
            f"max_level={self._max_level}"
            f")"
        )