from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence


Number = int | float


class EmbeddingModel(ABC):
    """
    Abstract interface for text embedding implementations.
    """

    @abstractmethod
    def embed(
        self,
        text: str,
    ) -> list[float]:
        """
        Generate an embedding for a single text.
        """
        raise NotImplementedError

    def embed_batch(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """
        Embed multiple texts.

        The default implementation calls `embed()` for each
        text. Concrete implementations can override this method
        to use efficient model-level batching.
        """

        if not isinstance(texts, Sequence):
            raise TypeError(
                "texts must be a sequence of strings."
            )

        return [
            self.embed(self._validate_text(text))
            for text in texts
        ]

    @staticmethod
    def _validate_text(text: str) -> str:
        """
        Validate and normalize input text.
        """

        if not isinstance(text, str):
            raise TypeError(
                "text must be a string."
            )

        text = text.strip()

        if not text:
            raise ValueError(
                "text must be non-empty."
            )

        return text

    @staticmethod
    def _validate_embedding(
        embedding: Sequence[Number],
    ) -> list[float]:
        """
        Validate and normalize an embedding vector.
        """

        if not isinstance(embedding, Sequence):
            raise TypeError(
                "embedding must be a sequence of numbers."
            )

        if len(embedding) == 0:
            raise ValueError(
                "embedding cannot be empty."
            )

        values: list[float] = []

        for value in embedding:

            if isinstance(value, bool):
                raise TypeError(
                    "embedding must contain only numeric values."
                )

            if not isinstance(value, (int, float)):
                raise TypeError(
                    "embedding must contain only numeric values."
                )

            value = float(value)

            if not math.isfinite(value):
                raise ValueError(
                    "embedding must contain only finite values."
                )

            values.append(value)

        return values

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"