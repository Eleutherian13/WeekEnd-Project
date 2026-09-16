from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class CrossEncoder(ABC):
    """
    Abstract interface for a cross-encoder relevance model.

    A cross-encoder receives the query and a candidate document together
    and produces a relevance score.

    Conceptually:

        query + document
              ↓
        cross-encoder
              ↓
        relevance score

    Unlike a bi-encoder, the query and document are evaluated jointly.
    """

    @abstractmethod
    def score(
        self,
        query: str,
        document: str,
    ) -> float:
        """
        Score the relevance of one document to a query.

        Parameters
        ----------
        query:
            User query.

        document:
            Textual representation of the candidate document.

        Returns
        -------
        float
            Relevance score assigned by the model.
        """

        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if not query.strip():
            raise ValueError("query must not be empty")

        if not isinstance(document, str):
            raise TypeError("document must be a string")

        if not document.strip():
            raise ValueError("document must not be empty")

        raise NotImplementedError

    def score_batch(
        self,
        query: str,
        documents: Sequence[str],
    ) -> Sequence[float]:
        """
        Score multiple documents for the same query.

        The default implementation calls score() individually.

        Concrete model adapters can override this method to use
        efficient batched inference.
        """

        if not isinstance(documents, Sequence):
            raise TypeError(
                "documents must be a sequence"
            )

        return [
            self.score(
                query=query,
                document=document,
            )
            for document in documents
        ]

    def __call__(
        self,
        query: str,
        document: str,
    ) -> float:
        """Allow the encoder to be called like a function."""
        return self.score(
            query=query,
            document=document,
        )