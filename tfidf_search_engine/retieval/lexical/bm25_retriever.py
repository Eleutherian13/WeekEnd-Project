
from __future__ import annotations 
from collections.abc import Iterable 

from ranking.bm25 import BM25 
from retieval.candidate_generator import CandidateGenerator 
from retieval.lexical.retriever import LexicalRetriever

class BM25Retriever(LexicalRetriever) :

    def __init__(self, candidate_generator : CandidateGenerator , scorer: BM25) -> None : 
        super().__init__(candidate_generator) 
        if not isinstance(scorer , BM25) :
            raise TypeError("scorer must be an instance of BM25.")
        self.scorer = scorer 

    def retrieve(self, query_terms : Iterable[str] ) -> list[tuple[int , float]] : 

        terms = list(query_terms) 

        if not terms : 
            return []

        candidates = self.candidate_generator.generate(terms)

        if not candidates : 
            return []

        scores : list[tuple[int , float]] = []

        for document_id in candidates:
            score = self.scorer.score_document(terms, document_id)
            scores.append((document_id, score))

        return sorted(
            scores,
            key=lambda result: (-result[1], result[0]),
        )

    def __repr__(self) -> str:
        return (
            f"BM25Retriever("
            f"candidate_generator={self.candidate_generator!r}, "
            f"scorer={self.scorer!r}"
            f")"
        )


    # this was all about the BM25 retriever 


    
    