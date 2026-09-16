from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .evidence import Evidence


@dataclass(frozen=True, slots=True)
class RAGContext:
    """
    Structured context supplied to the answer generator.
    """

    query: str
    text: str
    evidence: tuple[Evidence, ...]


class ContextBuilder:
    """
    Builds a deterministic context block from retrieved evidence.

    The builder is responsible only for formatting and context-size
    control. It does not generate an answer.
    """

    DEFAULT_SEPARATOR: Final[str] = "\n\n"

    def __init__(
        self,
        max_characters: int | None = None,
    ) -> None:
        if max_characters is not None:
            if not isinstance(max_characters, int):
                raise TypeError(
                    "max_characters must be an integer or None"
                )

            if isinstance(max_characters, bool):
                raise TypeError(
                    "max_characters must be an integer or None"
                )

            if max_characters <= 0:
                raise ValueError(
                    "max_characters must be greater than 0"
                )

        self._max_characters = max_characters

    @property
    def max_characters(self) -> int | None:
        """Return the configured context character limit."""
        return self._max_characters

    def build(
        self,
        query: str,
        evidence: tuple[Evidence, ...] | list[Evidence],
    ) -> RAGContext:
        """
        Build formatted RAG context from evidence.
        """

        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if not query.strip():
            raise ValueError("query must not be empty")

        if not isinstance(evidence, (tuple, list)):
            raise TypeError(
                "evidence must be a tuple or list"
            )

        for item in evidence:
            if not isinstance(item, Evidence):
                raise TypeError(
                    "evidence must contain Evidence objects"
                )

        sections: list[str] = []
        selected: list[Evidence] = []
        character_count = 0

        for index, item in enumerate(evidence, start=1):
            section = self._format_evidence(
                index=index,
                evidence=item,
            )

            additional_length = (
                len(section)
                if not sections
                else len(self.DEFAULT_SEPARATOR) + len(section)
            )

            if (
                self._max_characters is not None
                and character_count + additional_length
                > self._max_characters
            ):
                break

            sections.append(section)
            selected.append(item)

            character_count += additional_length

        context_text = self.DEFAULT_SEPARATOR.join(
            sections
        )

        return RAGContext(
            query=query,
            text=context_text,
            evidence=tuple(selected),
        )

    @staticmethod
    def _format_evidence(
        index: int,
        evidence: Evidence,
    ) -> str:
        """
        Format one evidence item with explicit provenance.
        """

        return (
            f"[SOURCE {index}]\n"
            f"document_id: {evidence.document_id}\n"
            f"retrieval_score: {evidence.retrieval_score}\n"
            f"text:\n{evidence.text}"
        )

    def __call__(
        self,
        query: str,
        evidence: tuple[Evidence, ...] | list[Evidence],
    ) -> RAGContext:
        """Allow the builder to be called like a function."""
        return self.build(
            query=query,
            evidence=evidence,
        )