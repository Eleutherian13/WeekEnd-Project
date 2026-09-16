from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, cast

from analysis.analyzer import Analyzer
from query.query import Query
from query.query_processing import QueryProcessor
from retieval.dense.retriever import DenseRetriever
from retieval.lexical.retriever import LexicalRetriever
from retieval.lexical.retriever import RetrievalResult

from .fusion import FusionStrategy


class Retriever(Protocol):
    """
    Structural interface for any component capable of retrieval.

    Concrete lexical, dense, and graph retrievers do not need to
    inherit from a common class. They only need to expose a compatible
    retrieve() method.
    """

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> Sequence[RetrievalResult]:
        ...


class HybridRetriever:
    """
    Combines multiple retrieval systems into one hybrid retriever.

    The hybrid retriever orchestrates independent retrieval systems
    and delegates result combination to a FusionStrategy.

    Typical architecture:

        Query
          |
          +----> Lexical Retriever
          |
          +----> Dense Retriever
          |
          +----> Graph Retriever
          |
          v
       FusionStrategy
          |
          v
      Hybrid Results
    """

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        fusion_strategy: FusionStrategy,
        query_processor: QueryProcessor | None = None,
    ) -> None:
        if not isinstance(retrievers, Sequence):
            raise TypeError(
                "retrievers must be a sequence"
            )

        if not retrievers:
            raise ValueError(
                "retrievers must not be empty"
            )

        if not isinstance(fusion_strategy, FusionStrategy):
            raise TypeError(
                "fusion_strategy must be a FusionStrategy"
            )

        for retriever in retrievers:
            if not callable(
                getattr(retriever, "retrieve", None)
            ):
                raise TypeError(
                    "each retriever must provide a "
                    "retrieve(query, top_k) method"
                )

        if query_processor is not None and not isinstance(
            query_processor,
            QueryProcessor,
        ):
            raise TypeError(
                "query_processor must be a QueryProcessor or None"
            )

        self._retrievers = tuple(retrievers)
        self._fusion_strategy = fusion_strategy
        self._query_processor = (
            query_processor
            if query_processor is not None
            else QueryProcessor(Analyzer())
        )

    @property
    def retrievers(self) -> tuple[Retriever, ...]:
        """Return the configured retrievers."""
        return self._retrievers

    @property
    def fusion_strategy(self) -> FusionStrategy:
        """Return the configured fusion strategy."""
        return self._fusion_strategy

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> Sequence[RetrievalResult]:
        """
        Retrieve and fuse results from all configured retrievers.
        """

        self._validate_query(
            query=query,
            top_k=top_k,
        )

        result_lists: list[Sequence[RetrievalResult]] = []

        for retriever in self._retrievers:
            results = self._retrieve(
                retriever=retriever,
                query=query,
                top_k=top_k,
            )

            result_lists.append(results)

        return self._fusion_strategy.fuse(
            result_lists=result_lists,
            top_k=top_k,
        )

    def _retrieve(
        self,
        retriever: Retriever,
        query: str,
        top_k: int,
    ) -> Sequence[RetrievalResult]:
        if isinstance(retriever, LexicalRetriever):
            terms = self._query_processor.process(Query(query))
            lexical_retriever = cast(
                LexicalRetriever,
                retriever,
            )
            lexical_results = lexical_retriever.retrieve(terms)

            return [
                self._tuple_to_result(result)
                for result in lexical_results[:top_k]
            ]

        if isinstance(retriever, DenseRetriever):
            dense_retriever = cast(
                DenseRetriever,
                retriever,
            )
            dense_results = dense_retriever.retrieve(
                query=query,
                k=top_k,
            )

            return [
                self._tuple_to_result(result)
                for result in dense_results
            ]

        results = retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        if not isinstance(results, Sequence):
            raise TypeError(
                "retriever results must be a sequence"
            )

        for result in results:
            if not isinstance(result, RetrievalResult):
                raise TypeError(
                    "retrievers must return RetrievalResult objects"
                )

        return results

    @staticmethod
    def _tuple_to_result(
        result: tuple[str | int, float],
    ) -> RetrievalResult:
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError(
                "tuple retrievers must return (document_id, score)"
            )

        document_id, score = result

        if not isinstance(document_id, (str, int)):
            raise TypeError("retrieval document IDs must be strings or integers")

        if isinstance(document_id, bool):
            raise TypeError("retrieval document IDs must be strings or integers")

        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise TypeError("retrieval scores must be numeric")

        return RetrievalResult(
            document_id=str(document_id),
            score=float(score),
        )

    def __call__(
        self,
        query: str,
        top_k: int = 10,
    ) -> Sequence[RetrievalResult]:
        """Allow the hybrid retriever to be called like a function."""
        return self.retrieve(
            query=query,
            top_k=top_k,
        )

    @staticmethod
    def _validate_query(
        query: str,
        top_k: int,
    ) -> None:
        """Validate the public retrieval arguments."""

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string"
            )

        if not query.strip():
            raise ValueError(
                "query must not be empty"
            )

        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise TypeError(
                "top_k must be an integer"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )