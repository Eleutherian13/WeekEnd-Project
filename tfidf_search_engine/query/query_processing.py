from analysis.analyzer import Analyzer
from query.query import Query


class QueryProcessor:
    """
    Converts a Query into analyzed search terms.
    """

    def __init__(self, analyzer: Analyzer) -> None:

        if not isinstance(analyzer, Analyzer):
            raise TypeError(
                "analyzer must be an Analyzer"
            )

        self.analyzer = analyzer

    def process(
        self,
        query: Query,
    ) -> list[str]:

        if not isinstance(query, Query):
            raise TypeError(
                "query must be a Query"
            )

        return self.analyzer(query.text)

    def __call__(
        self,
        query: Query,
    ) -> list[str]:

        return self.process(query)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"