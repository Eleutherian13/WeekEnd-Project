from query import Query


class QueryParser:

    def parse(self, text: str) -> Query:
        return Query(text)

    

