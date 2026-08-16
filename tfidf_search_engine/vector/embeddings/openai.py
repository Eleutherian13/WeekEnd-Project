from __future__ import annotations

from collections.abc import Sequence

from openai import OpenAI

from vector.embedding import EmbeddingModel


class OpenAIEmbedding(EmbeddingModel):
    """
    Embedding implementation backed by the OpenAI Embeddings API.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str | None = None,
    ) -> None:
        """
        Initialize the OpenAI embedding model.

        Parameters
        ----------
        model_name:
            Name of the OpenAI embedding model.

        api_key:
            Optional API key.

            If omitted, the OpenAI SDK will use the
            OPENAI_API_KEY environment variable.
        """

        if not isinstance(model_name, str):
            raise TypeError(
                "model_name must be a string."
            )

        model_name = model_name.strip()

        if not model_name:
            raise ValueError(
                "model_name must be non-empty."
            )

        if api_key is not None:

            if not isinstance(api_key, str):
                raise TypeError(
                    "api_key must be a string or None."
                )

            api_key = api_key.strip()

            if not api_key:
                raise ValueError(
                    "api_key must be non-empty when provided."
                )

        self.model_name = model_name

        self.client = (
            OpenAI(api_key=api_key)
            if api_key is not None
            else OpenAI()
        )

        self.dimension: int | None = None

    def embed(
        self,
        text: str,
    ) -> list[float]:
        """
        Generate an embedding for a single text.
        """

        text = self._validate_text(text)

        response = self.client.embeddings.create(
            model=self.model_name,
            input=text,
            encoding_format="float",
        )

        if not response.data:
            raise ValueError(
                "OpenAI embedding API returned no data."
            )

        if len(response.data) != 1:
            raise ValueError(
                "Expected exactly one embedding for a single input."
            )

        embedding = self._validate_embedding(
            response.data[0].embedding
        )

        self._update_dimension(embedding)

        return embedding

    def embed_batch(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Uses OpenAI's batch input support instead of repeatedly
        making individual embedding requests.
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

        response = self.client.embeddings.create(
            model=self.model_name,
            input=validated_texts,
            encoding_format="float",
        )

        if not response.data:
            raise ValueError(
                "OpenAI embedding API returned no data."
            )

        if len(response.data) != len(validated_texts):
            raise ValueError(
                "OpenAI embedding API returned an unexpected "
                "number of embeddings."
            )

        # The API response contains an index for each embedding.
        # Sort by index so output order always matches input order.
        ordered_data = sorted(
            response.data,
            key=lambda item: item.index,
        )

        validated_embeddings = [
            self._validate_embedding(
                item.embedding
            )
            for item in ordered_data
        ]

        for embedding in validated_embeddings:
            self._update_dimension(embedding)

        return validated_embeddings

    def _update_dimension(
        self,
        embedding: Sequence[float],
    ) -> None:
        """
        Establish the embedding dimension on the first successful
        embedding and enforce it for all subsequent embeddings.
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
                "OpenAI embedding model returned embeddings "
                "with inconsistent dimensions."
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_name={self.model_name!r}, "
            f"dimension={self.dimension}"
            f")"
        )