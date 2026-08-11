from query.query import Query


class QueryParser:
    """
    Converts raw query text into a Query object.
    """

    def parse(self, text: str) -> Query:

        if not isinstance(text, str):
            raise TypeError(
                "text must be a string"
            )

        return Query(text)

    def __call__(self, text: str) -> Query:
        return self.parse(text)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"