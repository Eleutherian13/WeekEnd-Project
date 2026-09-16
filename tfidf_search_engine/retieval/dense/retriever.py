from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vector.vector_store import VectorStore


class DenseRetriever:
    """
    Dense retriever built on top of VectorStore.

    The retriever accepts raw query text and delegates:
        1. Query embedding
        2. Vector similarity search
        3. Metadata filtering

    to the configured VectorStore.

    Returns ranked results in the form:

        (document_id, score)

    Higher scores indicate greater similarity for cosine
    and dot-product metrics.
    """

    def __init__(
        self,
        vector_store: VectorStore,
    ) -> None:

        if not isinstance(vector_store, VectorStore):
            raise TypeError(
                "vector_store must be an instance of VectorStore."
            )

        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        k: int = 5,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Retrieve the top-k documents semantically similar
        to the supplied query.

        Parameters
        ----------
        query:
            Raw query text.

        k:
            Maximum number of results to return.

        metadata_filter:
            Optional exact metadata filter.

        Returns
        -------
        list[tuple[str, float]]
            Ranked results represented as:

                (document_id, score)
        """

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string."
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "query must be non-empty."
            )

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

        results = self.vector_store.search_text(
            text=query,
            k=k,
            metadata_filter=metadata_filter,
        )

        return [
            (result.id, result.score)
            for result in results
        ]

    def __call__(
        self,
        query: str,
        k: int = 5,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Callable interface for dense retrieval.
        """

        return self.retrieve(
            query=query,
            k=k,
            metadata_filter=metadata_filter,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"vector_store={self.vector_store!r}"
            f")"
        )


# this is the easiest way to make DenseRetriever a singleton

