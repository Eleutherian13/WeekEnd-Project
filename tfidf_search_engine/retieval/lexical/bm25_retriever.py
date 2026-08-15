from __future__ import annotations 
from collections.abc import Iterable 

from ranking.bm25 import BM25 
from retieval.lexical.retriever import LexicalRetriever 
from retieval.candidate_generator import CandidateGenerator


class BM25Retriever(LexicalRetriever):

    def __init__(self, candidate_generator : CandidateGenerator , scorer : BM25 ) -> None : 

        super().__init__(candidate_generator)

        if not isinstance(scorer , BM25 ) : 
            raise TypeError("scorer must be an instance of BM25")

        
        self.scorer = scorer 

    def retrieve(self , query_terms : Iterable[str] ) -> list[tuple[int , float]] : 

        candidates = self.candidate_generator.generate(query_terms)

        if not candidates : 
            return []

        terms = list(query_terms)

        if not terms :
            return []

        scores = list(tuples[int , float]) = []


        for document_id in candidates : 

            score = self.scorer.score(document_id = document_id , terms = terms)

            scores.append((document_id , score))

        return sorted(scores , key = lambda x : x[1] , reverse = True)

    def __repr__(self) -> str:

        return (
            f"BM25Retriever("
            f"candidate_generator={self.candidate_generator!r}, "
            f"scorer={self.scorer!r}"
            f")"    
        )


    

