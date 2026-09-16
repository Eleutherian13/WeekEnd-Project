from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from retieval.lexical.retriever import RetrievalResult


class Retriever(Protocol):
    """
    Structural interface for a component that can retrieve documents.
    """

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> Sequence[RetrievalResult]:
        ...


class RAGRetriever:
    """
    Retrieval adapter used by the RAG pipeline.

    This class does not implement retrieval itself.

    It delegates retrieval to an existing retriever, which may be:

    - a lexical retriever
    - a dense retriever
    - a graph retriever
    - a HybridRetriever
    - another compatible retrieval component
    """

    def __init__(self, retriever: Retriever) -> None:
        if not callable(getattr(retriever, "retrieve", None)):
            raise TypeError(
                "retriever must provide a retrieve(query, top_k) method"
            )

        self._retriever = retriever

    @property
    def retriever(self) -> Retriever:
        """Return the underlying retriever."""
        return self._retriever

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> Sequence[RetrievalResult]:
        """
        Retrieve candidates for a RAG query.
        """

        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if not query.strip():
            raise ValueError("query must not be empty")

        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise TypeError("top_k must be an integer")

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        results = self._retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        if not isinstance(results, Sequence):
            raise TypeError(
                "retriever must return a sequence"
            )

        for result in results:
            if not isinstance(result, RetrievalResult):
                raise TypeError(
                    "retriever results must contain "
                    "RetrievalResult objects"
                )

        return results

    def __call__(
        self,
        query: str,
        top_k: int = 10,
    ) -> Sequence[RetrievalResult]:
        """Allow the adapter to be called like a function."""
        return self.retrieve(
            query=query,
            top_k=top_k,
        )