# `retrieval/lexical/retriever.py`

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from retieval.candidate_generator import CandidateGenerator


class LexicalRetriever(ABC):
    """
    Abstract base class for lexical retrieval.

    A lexical retriever takes already processed query terms,
    generates candidate documents, applies a lexical scoring
    strategy, and returns ranked retrieval results.

    Concrete subclasses are responsible for the actual scoring
    strategy, such as BM25 or TF-IDF.
    """

    def __init__(
        self,
        candidate_generator: CandidateGenerator,
    ) -> None:
        if not isinstance(candidate_generator, CandidateGenerator):
            raise TypeError(
                "candidate_generator must be an instance of "
                "CandidateGenerator."
            )

        self.candidate_generator = candidate_generator

    @abstractmethod
    def retrieve(
        self,
        query_terms: Iterable[str],
    ) -> list[tuple[int, float]]:
        """
        Retrieve and rank documents for the supplied query terms.

        Parameters
        ----------
        query_terms:
            Already processed query terms.

        Returns
        -------
        list[tuple[int, float]]
            Ranked results represented as:

                (document_id, score)

            Higher scores indicate greater relevance.

        Notes
        -----
        Concrete subclasses must implement the scoring and
        ranking strategy.
        """
        raise NotImplementedError

    def __call__(
        self,
        query_terms: Iterable[str],
    ) -> list[tuple[int, float]]:
        """
        Callable interface for lexical retrieval.
        """
        return self.retrieve(query_terms)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"candidate_generator={self.candidate_generator!r})"
        )

