from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeAlias

from retieval.lexical.retriever import RetrievalResult

from .cross_encoder import CrossEncoder


DocumentTextProvider: TypeAlias = Callable[[str], str]


class Reranker:
    """
    Reranks an existing candidate set using a cross-encoder.

    The reranker intentionally does not perform candidate retrieval.

    Its responsibility is:

        candidates
            ↓
        obtain document text
            ↓
        cross-encoder scoring
            ↓
        reorder candidates
            ↓
        return top-k
    """

    def __init__(
        self,
        cross_encoder: CrossEncoder,
        document_text_provider: DocumentTextProvider,
    ) -> None:
        if not isinstance(cross_encoder, CrossEncoder):
            raise TypeError(
                "cross_encoder must be a CrossEncoder"
            )

        if not callable(document_text_provider):
            raise TypeError(
                "document_text_provider must be callable"
            )

        self._cross_encoder = cross_encoder
        self._document_text_provider = document_text_provider

    @property
    def cross_encoder(self) -> CrossEncoder:
        """Return the configured cross-encoder."""
        return self._cross_encoder

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: int = 10,
    ) -> Sequence[RetrievalResult]:
        """
        Rerank retrieval candidates using the cross-encoder.

        Parameters
        ----------
        query:
            User query.

        candidates:
            Candidate documents produced by an earlier retrieval stage.

        top_k:
            Maximum number of reranked documents to return.

        Returns
        -------
        Sequence[RetrievalResult]
            Candidates ordered by cross-encoder relevance score.
        """

        self._validate_inputs(
            query=query,
            candidates=candidates,
            top_k=top_k,
        )

        if not candidates:
            return []

        unique_candidates = self._deduplicate_candidates(
            candidates
        )

        document_ids: list[str] = []
        documents: list[str] = []

        for candidate in unique_candidates:
            document_id = candidate.document_id
            document = self._document_text_provider(
                document_id
            )

            if not isinstance(document, str):
                raise TypeError(
                    "document_text_provider must return strings"
                )

            if not document.strip():
                raise ValueError(
                    f"document text is empty for "
                    f"document: {document_id}"
                )

            document_ids.append(document_id)
            documents.append(document)

        scores = self._cross_encoder.score_batch(
            query=query,
            documents=documents,
        )

        if len(scores) != len(document_ids):
            raise ValueError(
                "cross_encoder returned a different number "
                "of scores than candidate documents"
            )

        reranked: list[RetrievalResult] = []

        for candidate, score in zip(
            unique_candidates,
            scores,
            strict=True,
        ):
            if not isinstance(score, (int, float)):
                raise TypeError(
                    "cross_encoder scores must be numeric"
                )

            reranked.append(
                RetrievalResult(
                    document_id=candidate.document_id,
                    score=float(score),
                    metadata=candidate.metadata,
                )
            )

        reranked.sort(
            key=lambda result: (
                -result.score,
                result.document_id,
            )
        )

        return reranked[:top_k]

    def __call__(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: int = 10,
    ) -> Sequence[RetrievalResult]:
        """Allow the reranker to be called like a function."""
        return self.rerank(
            query=query,
            candidates=candidates,
            top_k=top_k,
        )

    @staticmethod
    def _deduplicate_candidates(
        candidates: Sequence[RetrievalResult],
    ) -> list[RetrievalResult]:
        """
        Remove duplicate document IDs while preserving the first
        occurrence.

        A document should only be sent to the expensive cross-encoder
        once.
        """

        seen: set[str] = set()
        unique: list[RetrievalResult] = []

        for candidate in candidates:
            if candidate.document_id in seen:
                continue

            seen.add(candidate.document_id)
            unique.append(candidate)

        return unique

    @staticmethod
    def _validate_inputs(
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: int,
    ) -> None:
        """Validate public reranking arguments."""

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string"
            )

        if not query.strip():
            raise ValueError(
                "query must not be empty"
            )

        if not isinstance(candidates, Sequence):
            raise TypeError(
                "candidates must be a sequence"
            )

        for candidate in candidates:
            if not isinstance(candidate, RetrievalResult):
                raise TypeError(
                    "candidates must contain "
                    "RetrievalResult objects"
                )

        if not isinstance(top_k, int):
            raise TypeError(
                "top_k must be an integer"
            )

        if isinstance(top_k, bool):
            raise TypeError(
                "top_k must be an integer"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )