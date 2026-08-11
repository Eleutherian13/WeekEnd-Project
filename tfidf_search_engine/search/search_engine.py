from query.query import Query 
from query.query_processing import QueryProcessor 
from retieval.candidate_generator import CandidateGenerator

class SearchEngine: 

    def __init__(self , query_processor : QueryProcessor , candidate_generator : CandidateGenerator) : 

        self.query_processor = query_processor 
        self.candidate_generator = candidate_generator 

    def search(self , text : str , mode : str ) -> set[int] : 

        query = Query(text)

        terms = self.query_processor.process(query)

        if mode == "AND" : 
            return self.candidate_generator.and_retrieve(terms)

        if mode == "OR" : 
            return self.candidate_generator.or_retrive(terms)

        raise ValueError(f"if unsupported retrieval then mode , {mode} ")

    def __repr__(self) -> str :
        return f"{self.__class__.__name__}()"


    




