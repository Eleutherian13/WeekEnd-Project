from __future__ import annotations

from query.query import Query
from query.query_processing import QueryProcessor
from retieval.lexical.retriever import LexicalRetriever


class SearchEngine:
    """Lexical search facade over the existing retriever contract."""

    def __init__(
        self,
        query_processor: QueryProcessor,
        lexical_retriever: LexicalRetriever,
    ) -> None:
        if not isinstance(query_processor, QueryProcessor):
            raise TypeError(
                "query_processor must be a QueryProcessor"
            )

        if not isinstance(lexical_retriever, LexicalRetriever):
            raise TypeError(
                "lexical_retriever must be a LexicalRetriever"
            )

        self.query_processor = query_processor
        self.lexical_retriever = lexical_retriever

    def search(
        self,
        text: str,
        mode: str = "OR",
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if not isinstance(mode, str):
            raise TypeError("mode must be a string")

        mode = mode.upper()
        if mode not in {"AND", "OR"}:
            raise ValueError(f"unsupported retrieval mode: {mode}")

        query = Query(text)
        terms = self.query_processor.process(query)
        results = self.lexical_retriever.retrieve(terms)

        if mode == "AND" and terms:
            candidate_generator = (
                self.lexical_retriever.candidate_generator
            )
            matching_documents = set.intersection(
                *(
                    candidate_generator.generate([term])
                    for term in terms
                )
            )
            results = [
                result
                for result in results
                if result[0] in matching_documents
            ]

        return results[:top_k]

    def __call__(
        self,
        text: str,
        mode: str = "OR",
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        return self.search(
            text=text,
            top_k=top_k,
            mode=mode,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"







