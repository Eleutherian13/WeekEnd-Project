from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

from sentence_transformers import CrossEncoder as SentenceTransformersModel


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


class SentenceTransformerCrossEncoder(CrossEncoder):
    """Cross-encoder adapter backed by Sentence Transformers."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        if not isinstance(model_name, str):
            raise TypeError("model_name must be a string")

        model_name = model_name.strip()
        if not model_name:
            raise ValueError("model_name must not be empty")

        self.model_name = model_name
        self.model = SentenceTransformersModel(model_name)

    def score(
        self,
        query: str,
        document: str,
    ) -> float:
        self._validate_text(query, "query")
        self._validate_text(document, "document")

        return self.score_batch(
            query=query,
            documents=[document],
        )[0]

    def score_batch(
        self,
        query: str,
        documents: Sequence[str],
    ) -> list[float]:
        self._validate_text(query, "query")

        if isinstance(documents, (str, bytes)):
            raise TypeError("documents must be a sequence")

        if not isinstance(documents, Sequence):
            raise TypeError("documents must be a sequence")

        for document in documents:
            self._validate_text(document, "document")

        if not documents:
            return []

        pairs = [(query, document) for document in documents]
        raw_scores = self.model.predict(pairs)

        if hasattr(raw_scores, "tolist"):
            raw_scores = raw_scores.tolist()

        if isinstance(raw_scores, (int, float)):
            raw_scores = [raw_scores]

        try:
            scores = list(raw_scores)
        except TypeError as error:
            raise TypeError(
                "cross-encoder must return a sequence of scores"
            ) from error

        if len(scores) != len(documents):
            raise ValueError(
                "cross-encoder returned a different number of scores "
                "than documents"
            )

        validated_scores: list[float] = []
        for score in scores:
            if isinstance(score, bool) or not isinstance(
                score,
                (int, float),
            ):
                raise TypeError("cross-encoder scores must be numeric")

            score = float(score)
            if not math.isfinite(score):
                raise ValueError(
                    "cross-encoder scores must be finite"
                )

            validated_scores.append(score)

        return validated_scores

    @staticmethod
    def _validate_text(value: str, name: str) -> None:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string")

        if not value.strip():
            raise ValueError(f"{name} must not be empty")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_name={self.model_name!r}"
            f")"
        )