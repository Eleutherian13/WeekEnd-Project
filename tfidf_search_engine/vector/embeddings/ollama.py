from __future__ import annotations
from collections.abc import Sequence
import ollama 
from vector.embedding import EmbeddingModel


class OllamaEmbedding(EmbeddingModel):

    def __init__(self , model_name : str = "ollama/distiluse-base-multilingual-cased" ) -> None :

        if not isinstance(model_name , str) : 
            raise TypeError("model_name must be a string.")

        model_name = model_name.strip()

        if not model_name : 
            raise ValueError("model_name must be non-empty.")

        self.model_name = model_name 

        self.dimension : int | None = None 

    def embed(self , text : str) -> list[float] : 

        text = self._validate_text(text)

        response = ollama.embed(text , model = self.model_name , input = text)

        embeddings = response.get("embedding")

        if not embeddings :
            raise ValueError("Embedding model returned no embeddings.")

        embedding = self._validate_embedding(embeddings[0])

        return embedding

    def embed_batch(self , texts : Sequence[str] ) -> list[list[float]] :

        if not isinstance(texts , Sequence ) : 

            raise TypeError("texts must be a sequence of strings.")

        if len(texts) == 0 :
            return []

        validate_texts = [
            self._validate_text(text)
            for text in texts
        ]

        response = ollama.embed(
            model = self.model_name ,
            input = validate_texts
        )

        embeddings = response.get("embedding")

        if not embeddings :
            raise ValueError("Embedding model returned no embeddings.")

        validated_embeddings = [
            self._validate_embedding(embedding)
            for embedding in embeddings
        ]

        for embedding in validated_embeddings : 
            if len(embedding) != self.dimension :
                raise ValueError("Embedding model returned embeddings with inconsistent dimensions.")
            
        # update dimension remaining abhi ke liye fir baad me likhunga 

        return validated_embeddings


    def _update_dimension(self , embedding : Sequence[float] ) -> None :
        dimension = len(embedding)

        if self.dimension is None :
            self.dimension = dimension
            return 

        if dimension != self.dimension : 
            raise ValueError("Embedding model returned embeddings with inconsistent dimensions.")

        return 

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_name={self.model_name!r}, "
            f"dimension={self.dimension}"
            f")"
        )


    

