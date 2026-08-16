from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any

from vector.embedding import EmbeddingModel
from vector.similarity import VectorSimilarity


Number = int | float


@dataclass(frozen=True)
class VectorSearchResult:
    """
    Result returned by VectorStore.search().
    """

    id: str
    score: float
    metadata: dict[str, Any]


class VectorStore:
    """
    In-memory vector store.

    Responsibilities
    ----------------
    - Store vectors using unique IDs.
    - Store optional metadata associated with vectors.
    - Enforce vector dimensionality.
    - Provide CRUD operations.
    - Perform exact nearest-neighbor search.
    - Optionally use an EmbeddingModel for text-based operations.

    HNSW is intentionally not handled here yet.
    HNSW will later become the approximate nearest-neighbor
    indexing layer on top of this storage abstraction.
    """

    _VALID_METRICS = {
        "cosine",
        "dot",
        "euclidean",
    }

    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        dimension: int | None = None,
        metric: str = "cosine",
    ) -> None:
        """
        Initialize the vector store.

        Parameters
        ----------
        embedding_model:
            Optional embedding model used by add_text() and
            search_text().

        dimension:
            Expected vector dimension.

            If omitted, the dimension is inferred from the
            first vector inserted.

        metric:
            Similarity/distance metric.

            Supported values:
                - cosine
                - dot
                - euclidean
        """

        if embedding_model is not None and not isinstance(
            embedding_model,
            EmbeddingModel,
        ):
            raise TypeError(
                "embedding_model must be an EmbeddingModel "
                "instance or None."
            )

        if dimension is not None:

            if isinstance(dimension, bool):
                raise TypeError(
                    "dimension must be an integer or None."
                )

            if not isinstance(dimension, int):
                raise TypeError(
                    "dimension must be an integer or None."
                )

            if dimension <= 0:
                raise ValueError(
                    "dimension must be greater than zero."
                )

        if not isinstance(metric, str):
            raise TypeError(
                "metric must be a string."
            )

        metric = metric.strip().lower()

        if metric not in self._VALID_METRICS:
            raise ValueError(
                f"Unsupported metric: {metric!r}. "
                f"Supported metrics are: "
                f"{sorted(self._VALID_METRICS)}."
            )

        self.embedding_model = embedding_model

        self.dimension = dimension

        self.metric = metric

        self._vectors: dict[str, list[float]] = {}

        self._metadata: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_id(
        vector_id: str,
    ) -> str:
        """
        Validate a vector ID.
        """

        if not isinstance(vector_id, str):
            raise TypeError(
                "vector_id must be a string."
            )

        vector_id = vector_id.strip()

        if not vector_id:
            raise ValueError(
                "vector_id must be non-empty."
            )

        return vector_id

    def _validate_vector(
        self,
        vector: Sequence[Number],
    ) -> list[float]:
        """
        Validate a vector and enforce store dimensionality.
        """

        values = VectorSimilarity._validate_vector(
            vector,
            name="vector",
        )

        if self.dimension is None:
            self.dimension = len(values)

        elif len(values) != self.dimension:
            raise ValueError(
                "Vector dimension does not match the "
                f"store dimension ({self.dimension})."
            )

        return values

    @staticmethod
    def _validate_metadata(
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """
        Validate and copy metadata.
        """

        if metadata is None:
            return {}

        if not isinstance(metadata, Mapping):
            raise TypeError(
                "metadata must be a mapping or None."
            )

        return dict(metadata)

    def _validate_k(
        self,
        k: int,
    ) -> int:
        """
        Validate the requested number of search results.
        """

        if isinstance(k, bool):
            raise TypeError(
                "k must be an integer."
            )

        if not isinstance(k, int):
            raise TypeError(
                "k must be an integer."
            )

        if k <= 0:
            raise ValueError(
                "k must be greater than zero."
            )

        return k

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def add(
        self,
        vector_id: str,
        vector: Sequence[Number],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Add a vector to the store.

        Duplicate IDs are rejected.

        Use update() when replacing an existing vector.
        """

        vector_id = self._validate_id(vector_id)

        if vector_id in self._vectors:
            raise ValueError(
                f"Vector with id {vector_id!r} already exists."
            )

        validated_vector = self._validate_vector(vector)

        validated_metadata = self._validate_metadata(
            metadata
        )

        self._vectors[vector_id] = validated_vector

        self._metadata[vector_id] = validated_metadata

    def add_batch(
        self,
        items: Sequence[
            tuple[
                str,
                Sequence[Number],
                Mapping[str, Any] | None,
            ]
        ],
    ) -> None:
        """
        Add multiple vectors.

        Each item must have the form:

            (
                vector_id,
                vector,
                metadata,
            )

        The operation is validated before mutating the store.
        """

        if not isinstance(items, Sequence):
            raise TypeError(
                "items must be a sequence."
            )

        if len(items) == 0:
            return

        validated_items: list[
            tuple[str, list[float], dict[str, Any]]
        ] = []

        batch_ids: set[str] = set()

        for item in items:

            if not isinstance(item, Sequence):
                raise TypeError(
                    "Each batch item must be a sequence."
                )

            if len(item) != 3:
                raise ValueError(
                    "Each batch item must contain "
                    "vector_id, vector, and metadata."
                )

            vector_id = self._validate_id(item[0])

            if vector_id in self._vectors:
                raise ValueError(
                    f"Vector with id {vector_id!r} "
                    "already exists."
                )

            if vector_id in batch_ids:
                raise ValueError(
                    f"Duplicate vector id {vector_id!r} "
                    "found in batch."
                )

            vector = self._validate_vector(item[1])

            metadata = self._validate_metadata(item[2])

            batch_ids.add(vector_id)

            validated_items.append(
                (
                    vector_id,
                    vector,
                    metadata,
                )
            )

        for vector_id, vector, metadata in validated_items:

            self._vectors[vector_id] = vector

            self._metadata[vector_id] = metadata

    # ------------------------------------------------------------------
    # Text insertion
    # ------------------------------------------------------------------

    def add_text(
        self,
        vector_id: str,
        text: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Embed text using the configured embedding model
        and store the resulting vector.
        """

        if self.embedding_model is None:
            raise RuntimeError(
                "No embedding model is configured."
            )

        text = self.embedding_model._validate_text(
            text
        )

        vector = self.embedding_model.embed(text)

        self.add(
            vector_id=vector_id,
            vector=vector,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_vector(
        self,
        vector_id: str,
    ) -> list[float]:
        """
        Return a copy of the stored vector.
        """

        vector_id = self._validate_id(vector_id)

        if vector_id not in self._vectors:
            raise KeyError(
                f"Vector with id {vector_id!r} does not exist."
            )

        return list(self._vectors[vector_id])

    def get_metadata(
        self,
        vector_id: str,
    ) -> dict[str, Any]:
        """
        Return a copy of stored metadata.
        """

        vector_id = self._validate_id(vector_id)

        if vector_id not in self._metadata:
            raise KeyError(
                f"Vector with id {vector_id!r} does not exist."
            )

        return dict(self._metadata[vector_id])

    def get(
        self,
        vector_id: str,
    ) -> tuple[list[float], dict[str, Any]]:
        """
        Return both vector and metadata.
        """

        return (
            self.get_vector(vector_id),
            self.get_metadata(vector_id),
        )

    def contains(
        self,
        vector_id: str,
    ) -> bool:
        """
        Return True if the vector ID exists.
        """

        vector_id = self._validate_id(vector_id)

        return vector_id in self._vectors

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        vector_id: str,
        vector: Sequence[Number] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Update an existing vector and/or its metadata.

        At least one of vector or metadata must be supplied.
        """

        vector_id = self._validate_id(vector_id)

        if vector_id not in self._vectors:
            raise KeyError(
                f"Vector with id {vector_id!r} does not exist."
            )

        if vector is None and metadata is None:
            raise ValueError(
                "At least one of vector or metadata "
                "must be provided."
            )

        if vector is not None:

            validated_vector = self._validate_vector(
                vector
            )

            self._vectors[vector_id] = validated_vector

        if metadata is not None:

            validated_metadata = self._validate_metadata(
                metadata
            )

            self._metadata[vector_id] = validated_metadata

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(
        self,
        vector_id: str,
    ) -> None:
        """
        Delete a vector and its metadata.
        """

        vector_id = self._validate_id(vector_id)

        if vector_id not in self._vectors:
            raise KeyError(
                f"Vector with id {vector_id!r} does not exist."
            )

        del self._vectors[vector_id]

        del self._metadata[vector_id]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: Sequence[Number],
        k: int = 5,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """
        Perform exact nearest-neighbor search.

        Every stored vector is compared against the query vector.

        For cosine and dot product:
            higher scores are better.

        For Euclidean distance:
            lower scores are better.
        """

        k = self._validate_k(k)

        query = self._validate_vector(
            query_vector
        )

        metadata_filter = self._validate_metadata(
            metadata_filter
        )

        results: list[VectorSearchResult] = []

        for vector_id, vector in self._vectors.items():

            metadata = self._metadata[vector_id]

            if not self._matches_filter(
                metadata,
                metadata_filter,
            ):
                continue

            score = self._calculate_score(
                query,
                vector,
            )

            results.append(
                VectorSearchResult(
                    id=vector_id,
                    score=score,
                    metadata=dict(metadata),
                )
            )

        if self.metric == "euclidean":

            results.sort(
                key=lambda result: result.score
            )

        else:

            results.sort(
                key=lambda result: result.score,
                reverse=True,
            )

        return results[:k]

    def search_text(
        self,
        text: str,
        k: int = 5,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """
        Embed text using the configured embedding model
        and perform vector search.
        """

        if self.embedding_model is None:
            raise RuntimeError(
                "No embedding model is configured."
            )

        text = self.embedding_model._validate_text(
            text
        )

        query_vector = self.embedding_model.embed(text)

        return self.search(
            query_vector=query_vector,
            k=k,
            metadata_filter=metadata_filter,
        )

    def _calculate_score(
        self,
        query_vector: Sequence[Number],
        stored_vector: Sequence[Number],
    ) -> float:
        """
        Calculate the score according to the configured metric.
        """

        if self.metric == "cosine":

            return VectorSimilarity.cosine_similarity(
                query_vector,
                stored_vector,
            )

        if self.metric == "dot":

            return VectorSimilarity.dot_product(
                query_vector,
                stored_vector,
            )

        if self.metric == "euclidean":

            return VectorSimilarity.euclidean_distance(
                query_vector,
                stored_vector,
            )

        raise RuntimeError(
            f"Unsupported metric: {self.metric!r}."
        )

    @staticmethod
    def _matches_filter(
        metadata: Mapping[str, Any],
        metadata_filter: Mapping[str, Any],
    ) -> bool:
        """
        Perform simple exact metadata filtering.

        Every key/value pair in metadata_filter must match
        the stored metadata.
        """

        if not metadata_filter:
            return True

        return all(
            metadata.get(key) == value
            for key, value in metadata_filter.items()
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count(self) -> int:
        """
        Return the number of stored vectors.
        """

        return len(self._vectors)

    def clear(self) -> None:
        """
        Remove all vectors and metadata.

        The configured dimension is preserved.
        """

        self._vectors.clear()

        self._metadata.clear()

    def __len__(self) -> int:
        return self.count()

    def __contains__(
        self,
        vector_id: str,
    ) -> bool:
        return self.contains(vector_id)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"count={self.count()}, "
            f"dimension={self.dimension}, "
            f"metric={self.metric!r}"
            f")"
        )