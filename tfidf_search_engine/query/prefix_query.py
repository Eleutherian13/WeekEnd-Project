from index.vocabulary import Vocabulary
from index.inverted_index import InvertedIndex


class PrefixQuery:
    """
    Matches vocabulary terms that start with a given prefix.

    Example:

        PrefixQuery("mach")

    can match:

        machine
        machinery
        machines
    """

    def __init__(self, prefix: str) -> None:

        if not isinstance(prefix, str):
            raise TypeError(
                "prefix must be a string"
            )

        if not prefix:
            raise ValueError(
                "prefix cannot be empty"
            )

        self.prefix = prefix

    def matching_terms(
        self,
        vocabulary: Vocabulary,
    ) -> list[str]:

        if not isinstance(
            vocabulary,
            Vocabulary,
        ):
            raise TypeError(
                "vocabulary must be a Vocabulary"
            )

        return [
            term
            for term in vocabulary
            if term.startswith(self.prefix)
        ]

    def evaluate(
        self,
        vocabulary: Vocabulary,
        inverted_index: InvertedIndex,
    ) -> set[int]:

        if not isinstance(
            vocabulary,
            Vocabulary,
        ):
            raise TypeError(
                "vocabulary must be a Vocabulary"
            )

        if not isinstance(
            inverted_index,
            InvertedIndex,
        ):
            raise TypeError(
                "inverted_index must be an InvertedIndex"
            )

        matching_terms = self.matching_terms(
            vocabulary
        )

        documents = set()

        for term in matching_terms:

            posting_list = inverted_index.get(term)

            if posting_list is None:
                continue

            for posting in posting_list:
                documents.add(posting.document_id)

        return documents

    def __repr__(self) -> str:

        return (
            f"{self.__class__.__name__}"
            f"({self.prefix!r})"
        )