from __future__ import annotations

from collections.abc import Sequence

from sentence_transformers import SentenceTransformer

from vector.embedding import EmbeddingModel


class SentenceTransformerEmbedding(EmbeddingModel):
    """
    Embedding implementation backed by SentenceTransformers.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
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

        self.model = SentenceTransformer(
            self.model_name
        )

        dimension = (
            self.model
            .get_sentence_embedding_dimension()
        )

        if dimension is None:
            raise ValueError(
                "Unable to determine embedding dimension "
                "for the selected model."
            )

        if dimension <= 0:
            raise ValueError(
                "Embedding dimension must be greater than zero."
            )

        self.dimension = int(dimension)

    def embed(
        self,
        text: str,
    ) -> list[float]:
        """
        Generate an embedding for a single text.
        """

        text = self._validate_text(text)

        embedding = self.model.encode(
            text,
            convert_to_numpy=False,
        )

        embedding = self._validate_embedding(
            embedding
        )

        self._validate_dimension(embedding)

        return embedding

    def embed_batch(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts using
        SentenceTransformers batching.
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

        embeddings = self.model.encode(
            validated_texts,
            convert_to_numpy=False,
        )

        validated_embeddings = [
            self._validate_embedding(embedding)
            for embedding in embeddings
        ]

        for embedding in validated_embeddings:
            self._validate_dimension(embedding)

        if len(validated_embeddings) != len(validated_texts):
            raise ValueError(
                "Embedding model returned an unexpected "
                "number of embeddings."
            )

        return validated_embeddings

    def _validate_dimension(
        self,
        embedding: Sequence[float],
    ) -> None:
        """
        Ensure the returned embedding has the expected
        model dimension.
        """

        if len(embedding) != self.dimension:
            raise ValueError(
                "Embedding model returned an embedding "
                "with an unexpected dimension."
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_name={self.model_name!r}, "
            f"dimension={self.dimension}"
            f")"
        )