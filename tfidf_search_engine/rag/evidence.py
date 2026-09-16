from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, TypeAlias

from retieval.lexical.retriever import RetrievalResult


DocumentTextProvider: TypeAlias = Callable[[str], str]


@dataclass(frozen=True, slots=True)
class Evidence:
    """
    A piece of source evidence supplied to the RAG pipeline.

    Evidence preserves provenance from the original document through
    retrieval and into generation.
    """

    document_id: str
    text: str
    retrieval_score: float
    metadata: dict | None = None


class EvidenceBuilder:
    """
    Converts retrieval results into explicit evidence objects.

    This class is responsible for resolving document IDs into their
    actual textual content.
    """

    def __init__(
        self,
        document_text_provider: DocumentTextProvider,
    ) -> None:
        if not callable(document_text_provider):
            raise TypeError(
                "document_text_provider must be callable"
            )

        self._document_text_provider = document_text_provider

    def build(
        self,
        results: Sequence[RetrievalResult],
    ) -> Sequence[Evidence]:
        """
        Build evidence from retrieval results.
        """

        if not isinstance(results, Sequence):
            raise TypeError(
                "results must be a sequence"
            )

        evidence: list[Evidence] = []
        seen: set[str] = set()

        for result in results:
            if not isinstance(result, RetrievalResult):
                raise TypeError(
                    "results must contain RetrievalResult objects"
                )

            if result.document_id in seen:
                continue

            seen.add(result.document_id)

            text = self._document_text_provider(
                result.document_id
            )

            if not isinstance(text, str):
                raise TypeError(
                    "document_text_provider must return strings"
                )

            if not text.strip():
                raise ValueError(
                    f"document text is empty for document "
                    f"{result.document_id!r}"
                )

            evidence.append(
                Evidence(
                    document_id=result.document_id,
                    text=text,
                    retrieval_score=result.score,
                    metadata=result.metadata,
                )
            )

        return evidence

    def __call__(
        self,
        results: Sequence[RetrievalResult],
    ) -> Sequence[Evidence]:
        """Allow the builder to be called like a function."""
        return self.build(results)