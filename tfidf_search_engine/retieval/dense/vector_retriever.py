from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from retieval.dense.retriever import DenseRetriever
from retieval.lexical.retriever import RetrievalResult
from vector.vector_store import VectorStore


class VectorRetriever(DenseRetriever):
    """
    Concrete dense retriever backed by VectorStore.

    VectorRetriever accepts raw query text and delegates the actual
    embedding and vector search operations to VectorStore.

    Flow:

        Query Text
            ↓
        VectorRetriever
            ↓
        VectorStore.search_text()
            ↓
        EmbeddingModel
            ↓
        Vector Similarity Search
            ↓
        Ranked Results
    """

    def __init__(self, vector_store: VectorStore) -> None:
        super().__init__(vector_store)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        return super().retrieve(
            query=query,
            k=k,
            metadata_filter=metadata_filter,
        )

    def __repr__(self) -> str:
        return (
            f"VectorRetriever("
            f"vector_store={self.vector_store!r}"
            f")"
        )