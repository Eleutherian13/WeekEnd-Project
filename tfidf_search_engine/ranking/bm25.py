import math


from index.forward_index import ForwardIndex
from index.statistics import Statistics


class BM25:
    """
    Calculates BM25 relevance scores for
    query terms and indexed documents.
    """

    def __init__(
        self,
        forward_index: ForwardIndex,
        statistics: Statistics,
        k1: float = 1.5,
        b: float = 0.75,
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

        if not isinstance(k1, float):
            raise TypeError(
                "k1 must be a float"
            )

        if isinstance(k1, bool):
            raise TypeError(
                "k1 must be a float"
            )

        if k1 < 0.0:
            raise ValueError(
                "k1 must be non-negative"
            )

        if not isinstance(b, float):
            raise TypeError(
                "b must be a float"
            )

        if isinstance(b, bool):
            raise TypeError(
                "b must be a float"
            )

        if not 0.0 <= b <= 1.0:
            raise ValueError(
                "b must be between 0.0 and 1.0"
            )

        self.forward_index = forward_index
        self.statistics = statistics

        self.k1 = k1
        self.b = b

    def inverse_document_frequency(
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
            document_count
            - document_frequency
            + 0.5
        ) / (
            document_frequency
            + 0.5
        )

        return math.log(
            1.0 + ratio
        )

    def term_frequency_component(
        self,
        term_frequency: int,
        document_length: int,
        average_document_length: float,
    ) -> float:

        if not isinstance(
            term_frequency,
            int,
        ):
            raise TypeError(
                "term_frequency must be an integer"
            )

        if isinstance(
            term_frequency,
            bool,
        ):
            raise TypeError(
                "term_frequency must be an integer"
            )

        if not isinstance(
            document_length,
            int,
        ):
            raise TypeError(
                "document_length must be an integer"
            )

        if isinstance(
            document_length,
            bool,
        ):
            raise TypeError(
                "document_length must be an integer"
            )

        if not isinstance(
            average_document_length,
            float,
        ):
            raise TypeError(
                "average_document_length must be a float"
            )

        if isinstance(
            average_document_length,
            bool,
        ):
            raise TypeError(
                "average_document_length must be a float"
            )

        if term_frequency < 0:
            raise ValueError(
                "term_frequency must be non-negative"
            )

        if document_length < 0:
            raise ValueError(
                "document_length must be non-negative"
            )

        if average_document_length <= 0.0:
            return 0.0

        if term_frequency == 0:
            return 0.0

        normalization = (
            1.0
            - self.b
            + self.b
            * (
                document_length
                / average_document_length
            )
        )

        numerator = (
            term_frequency
            * (self.k1 + 1.0)
        )

        denominator = (
            term_frequency
            + self.k1 * normalization
        )

        return numerator / denominator

    def score(
        self,
        term: str,
        document_id: int,
    ) -> float:

        if not isinstance(term, str):
            raise TypeError(
                "term must be a string"
            )

        if not isinstance(
            document_id,
            int,
        ):
            raise TypeError(
                "document_id must be an integer"
            )

        if isinstance(
            document_id,
            bool,
        ):
            raise TypeError(
                "document_id must be an integer"
            )

        if document_id < 0:
            raise ValueError(
                "document_id must be non-negative"
            )

        term_frequency = (
            self.forward_index.get_term_frequency(
                document_id,
                term,
            )
        )

        if term_frequency == 0:
            return 0.0

        document_length = (
            self.forward_index.get_document_length(
                document_id
            )
        )

        average_document_length = (
            self.statistics.average_document_length()
        )

        idf = (
            self.inverse_document_frequency(
                term
            )
        )

        tf_component = (
            self.term_frequency_component(
                term_frequency,
                document_length,
                average_document_length,
            )
        )

        return idf * tf_component

    def score_document(
        self,
        terms: list[str],
        document_id: int,
    ) -> float:

        if not isinstance(terms, list):
            raise TypeError(
                "terms must be a list"
            )

        if not all(
            isinstance(term, str)
            for term in terms
        ):
            raise TypeError(
                "terms must contain only strings"
            )

        if not isinstance(
            document_id,
            int,
        ):
            raise TypeError(
                "document_id must be an integer"
            )

        if isinstance(
            document_id,
            bool,
        ):
            raise TypeError(
                "document_id must be an integer"
            )

        if document_id < 0:
            raise ValueError(
                "document_id must be non-negative"
            )

        total_score = 0.0

        for term in terms:

            total_score += self.score(
                term,
                document_id,
            )

        return total_score

    def __call__(
        self,
        terms: list[str],
        document_id: int,
    ) -> float:

        return self.score_document(
            terms,
            document_id,
        )

    def __repr__(self) -> str:

        return (
            f"{self.__class__.__name__}("
            f"forward_index="
            f"{self.forward_index!r}, "
            f"statistics="
            f"{self.statistics!r}, "
            f"k1={self.k1!r}, "
            f"b={self.b!r}"
            f")"
        )