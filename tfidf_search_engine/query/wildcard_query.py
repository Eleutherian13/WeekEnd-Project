import re

from index.vocabulary import Vocabulary
from index.inverted_index import InvertedIndex


class WildcardQuery:
    """
    Matches vocabulary terms using wildcard patterns.

    Supported wildcards:

        *  -> zero or more characters
        ?  -> exactly one character

    Examples:

        mach*
        *ing
        m?chine
    """

    def __init__(self, pattern: str) -> None:

        if not isinstance(pattern, str):
            raise TypeError(
                "pattern must be a string"
            )

        if not pattern:
            raise ValueError(
                "pattern cannot be empty"
            )

        self.pattern = pattern

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

        regex_pattern = self._compile_pattern()

        return [
            term
            for term in vocabulary
            if regex_pattern.fullmatch(term)
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
                documents.add(
                    posting.document_id
                )

        return documents

    def _compile_pattern(self) -> re.Pattern:

        regex = ""

        for character in self.pattern:

            if character == "*":
                regex += ".*"

            elif character == "?":
                regex += "."

            else:
                regex += re.escape(character)

        return re.compile(regex)

    def __repr__(self) -> str:

        return (
            f"{self.__class__.__name__}"
            f"({self.pattern!r})"
        )

    