# retrieval/lexical/retriever.py

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from retieval.candidate_generator import CandidateGenerator


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Ranked result shared by result-list retrieval components."""

    document_id: str
    score: float
    metadata: dict[str, Any] | None = None


class LexicalRetriever(ABC):
    """
    Abstract base class for lexical retrieval.

    A lexical retriever:
        1. Receives processed query terms.
        2. Generates candidate documents.
        3. Scores the candidate documents using a lexical
           ranking algorithm.
        4. Returns ranked document IDs with their scores.

    Concrete implementations can use ranking strategies
    such as BM25 or TF-IDF.
    """

    def __init__(
        self,
        candidate_generator: CandidateGenerator,
    ) -> None:

        if not isinstance(candidate_generator, CandidateGenerator):
            raise TypeError(
                "candidate_generator must be an instance of "
                "CandidateGenerator"
            )

        self.candidate_generator = candidate_generator

    @abstractmethod
    def retrieve(
        self,
        query_terms: Iterable[str],
    ) -> list[tuple[int, float]]:
        """
        Retrieve and rank documents for the given query terms.

        Parameters
        ----------
        query_terms:
            Already processed query terms.

        Returns
        -------
        list[tuple[int, float]]
            Ranked retrieval results in the form:

                (document_id, score)

            Results should be ordered from highest score
            to lowest score.
        """

        raise NotImplementedError

    def __call__(
        self,
        query_terms: Iterable[str],
    ) -> list[tuple[int, float]]:
        """
        Allow the retriever to be called like a function.

        Example
        -------
        retriever(["machine", "learning"])

        is equivalent to:

        retriever.retrieve(["machine", "learning"])
        """

        return self.retrieve(query_terms)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"candidate_generator={self.candidate_generator!r}"
            f")"
        )

    