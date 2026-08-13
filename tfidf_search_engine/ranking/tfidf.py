import math

from index.forward_index import ForwardIndex
from index.statistics import Statistics


class TFIDF:

    def __init__(
        self,
        forward_index: ForwardIndex,
        statistics: Statistics,
    ) -> None:

        if not isinstance(
            forward_index,
            ForwardIndex,
        ):
            raise TypeError(
                "forward_index must be a ForwardIndex"
            )

        if not isinstance(
            statistics,
            Statistics,
        ):
            raise TypeError(
                "statistics must be a Statistics"
            )

        self.forward_index = forward_index
        self.statistics = statistics

    def tf(
        self,
        term: str,
        document_id: int,
    ) -> int:

        if not isinstance(term, str):
            raise TypeError(
                "term must be a string"
            )

        if not isinstance(document_id, int):
            raise TypeError(
                "document_id must be an integer"
            )

        if document_id < 0:
            raise ValueError(
                "document_id must be a non-negative integer"
            )

        return self.forward_index.get_term_frequency(
            document_id,
            term,
        )

    def idf(
        self,
        term: str,
    ) -> float:

        if not isinstance(term, str):
            raise TypeError(
                "term must be a string"
            )

        document_count = (
            self.statistics.document_count()
        )

        document_frequency = (
            self.statistics.document_frequency(
                term
            )
        )

        ratio = (
            (document_count + 1)
            / (document_frequency + 1)
        )

        return math.log(ratio) + 1

    def tfidf(
        self,
        term: str,
        document_id: int,
    ) -> float:

        if not isinstance(term, str):
            raise TypeError(
                "term must be a string"
            )

        if not isinstance(document_id, int):
            raise TypeError(
                "document_id must be an integer"
            )

        if document_id < 0:
            raise ValueError(
                "document_id must be a non-negative integer"
            )

        return (
            self.tf(term, document_id)
            * self.idf(term)
        )

    def __call__(
        self,
        term: str,
        document_id: int,
    ) -> float:

        return self.tfidf(
            term,
            document_id,
        )

    def __repr__(self) -> str:

        return (
            f"{self.__class__.__name__}()"
        )