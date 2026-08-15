from __future__ import annotations
from collections.abc import Sequence
from sentence_transformers import SentenceTransformer
from vector.embedding import EmbeddingModel


class SentenceTransformerEmbedding(EmbeddingModel):


    def __init__(self , model_name : str = "sentence-transformers/all-MiniLM-L6-v2" ) ->None :

        if not isinstance(model_name , str) :
            raise TypeError("model_name must be a string.")

        model_name = model_name.strip()

        if not model_name :
            raise ValueError("model_name must be non-empty.")

        self.model_name = model_name

        self.model = SentenceTransformer(self.model_name)

        self.dimension = self.model.get_sentence_embedding_dimension()

        if self.dimension == 0 :
            raise ValueError("model_name must be a valid sentence transformer model.")

        if self.dimension is None : 
            raise ValueError("model_name must be a valid sentence transformer model.")

        self.dimension = int(self.dimension)

    def embed(self , text : str ) -> list[float] : 

        text = self._validate_text(text)

        embedding = self.model.encode(text , convert_to_numpy = False)

        return self._validate_embedding(embedding)


    def embed_batch(self , texts : Sequence[str] ) -> list[list[float]] :

        if not isinstance(texts , Sequence) :
            raise TypeError("texts must be a sequence of strings.")

        if len(texts) == 0 :

            return []

        embeddings = self.model.encode(texts , convert_to_numpy = False)

        return [
            self._validate_embedding(embedding) 
            for embedding in embeddings
        ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_name={self.model_name!r}, "
            f"dimension={self.dimension}"
            f")"
        )

        



