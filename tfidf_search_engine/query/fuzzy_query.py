from index.vocabulary import Vocabulary
from index.inverted_index import InvertedIndex


class FuzzyQuery:
    """
    Performs fuzzy matching using Levenshtein distance.

    Example:

        machne

    can match:

        machine

    when max_distance >= 1.
    """

    def __init__(
        self,
        term: str,
        max_distance: int = 1,
    ) -> None:

        if not isinstance(term, str):
            raise TypeError(
                "term must be a string"
            )

        if not term:
            raise ValueError(
                "term cannot be empty"
            )

        if not isinstance(max_distance, int):
            raise TypeError(
                "max_distance must be an integer"
            )

        if max_distance < 0:
            raise ValueError(
                "max_distance cannot be negative"
            )

        self.term = term
        self.max_distance = max_distance

    def distance(
        self,
        first: str,
        second: str,
    ) -> int:
        """
        Calculate Levenshtein distance between
        two strings.
        """

        if not isinstance(first, str):
            raise TypeError(
                "first must be a string"
            )

        if not isinstance(second, str):
            raise TypeError(
                "second must be a string"
            )

        # --------------------------------------------------
        # Optimization:
        # Make second the shorter string.
        # --------------------------------------------------

        if len(first) < len(second):
            first, second = second, first

        # --------------------------------------------------
        # Empty string cases
        # --------------------------------------------------

        if len(second) == 0:
            return len(first)

        # --------------------------------------------------
        # Previous row of the DP matrix
        # --------------------------------------------------

        previous_row = list(
            range(len(second) + 1)
        )

        # --------------------------------------------------
        # Build the matrix row by row
        # --------------------------------------------------

        for i, first_character in enumerate(
            first,
            start=1,
        ):

            current_row = [i]

            for j, second_character in enumerate(
                second,
                start=1,
            ):

                insertion = (
                    current_row[j - 1] + 1
                )

                deletion = (
                    previous_row[j] + 1
                )

                substitution = (
                    previous_row[j - 1]
                    + (
                        first_character
                        != second_character
                    )
                )

                current_row.append(
                    min(
                        insertion,
                        deletion,
                        substitution,
                    )
                )

            previous_row = current_row

        return previous_row[-1]

    def matching_terms(
        self,
        vocabulary: Vocabulary,
    ) -> list[str]:
        """
        Return vocabulary terms whose edit distance
        from the query term is within max_distance.
        """

        if not isinstance(
            vocabulary,
            Vocabulary,
        ):
            raise TypeError(
                "vocabulary must be a Vocabulary"
            )

        matches = []

        for vocabulary_term in vocabulary:

            distance = self.distance(
                self.term,
                vocabulary_term,
            )

            if distance <= self.max_distance:
                matches.append(
                    vocabulary_term
                )

        return matches

    def evaluate(
        self,
        vocabulary: Vocabulary,
        inverted_index: InvertedIndex,
    ) -> set[int]:
        """
        Find all documents containing terms that
        approximately match the query term.
        """

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

    def __repr__(self) -> str:

        return (
            f"{self.__class__.__name__}("
            f"{self.term!r}, "
            f"max_distance="
            f"{self.max_distance!r}"
            f")"
        )


    