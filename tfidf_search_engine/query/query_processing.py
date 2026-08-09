from analysis.analyzer import Analyzer
# Inside query_processing.py
from query.query import Query

class QueryProcessor :

    def __init__(self , analyzer : Analyzer )  : 

        self.analyzer = analyzer 


    def process(self , query : Query ) -> list[str] : 

        return self.analyzer(query.text)

    def __call__(self , query : Query ) -> list[str] :
        return self.process(query)

if __name__ == "__main__" :
    analyzer = Analyzer()
    processor = QueryProcessor(analyzer)
    query = Query("Machine Learning!!")
    terms = processor.process(query)
    print("Raw query:", query.text)

