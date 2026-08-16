from __future__ import annotations

from collections.abc import Sequence

import ollama

from vector.embedding import EmbeddingModel


class OllamaEmbedding(EmbeddingModel):
    """
    Embedding implementation backed by Ollama.
    """

    def __init__(
        self,
        model_name: str = "all-minilm",
    ) -> None:

        if not isinstance(model_name, str):
            raise TypeError(
                "model_name must be a string."
            )

        model_name = model_name.strip()

        if not model_name:
            raise ValueError(
                "model_name must be non-empty."
            )

        self.model_name = model_name

        self.dimension: int | None = None

    def embed(
        self,
        text: str,
    ) -> list[float]:
        """
        Generate an embedding for a single text.
        """

        text = self._validate_text(text)

        response = ollama.embed(
            model=self.model_name,
            input=text,
        )

        embeddings = response["embeddings"]

        if not embeddings:
            raise ValueError(
                "Embedding model returned no embeddings."
            )

        if len(embeddings) != 1:
            raise ValueError(
                "Expected exactly one embedding for a single input."
            )

        embedding = self._validate_embedding(
            embeddings[0]
        )

        self._update_dimension(embedding)

        return embedding

    def embed_batch(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts using
        Ollama's batch embedding API.
        """

        if not isinstance(texts, Sequence):
            raise TypeError(
                "texts must be a sequence of strings."
            )

        if len(texts) == 0:
            return []

        validated_texts = [
            self._validate_text(text)
            for text in texts
        ]

        response = ollama.embed(
            model=self.model_name,
            input=validated_texts,
        )

        embeddings = response["embeddings"]

        if not embeddings:
            raise ValueError(
                "Embedding model returned no embeddings."
            )

        if len(embeddings) != len(validated_texts):
            raise ValueError(
                "Embedding model returned an unexpected "
                "number of embeddings."
            )

        validated_embeddings = [
            self._validate_embedding(embedding)
            for embedding in embeddings
        ]

        for embedding in validated_embeddings:
            self._update_dimension(embedding)

        return validated_embeddings

    def _update_dimension(
        self,
        embedding: Sequence[float],
    ) -> None:
        """
        Establish the embedding dimension on the first
        successful embedding and enforce it thereafter.
        """

        dimension = len(embedding)

        if dimension <= 0:
            raise ValueError(
                "Embedding dimension must be greater than zero."
            )

        if self.dimension is None:
            self.dimension = dimension
            return

        if dimension != self.dimension:
            raise ValueError(
                "Embedding model returned embeddings with "
                "inconsistent dimensions."
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_name={self.model_name!r}, "
            f"dimension={self.dimension}"
            f")"
        )