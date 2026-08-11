from index.positional_index import PositionalIndex


class PhraseQuery:
    """
    Represents a phrase query.

    Example:

        "machine learning"

    Internally:

        ["machine", "learning"]

    A document matches only when the terms occur
    consecutively and in the correct order.
    """

    def __init__(self, terms: list[str]) -> None:

        if not isinstance(terms, list):
            raise TypeError("terms must be a list")

        if not terms:
            raise ValueError(
                "phrase must contain at least one term"
            )

        if not all(
            isinstance(term, str)
            for term in terms
        ):
            raise TypeError(
                "all terms must be strings"
            )

        if not all(terms):
            raise ValueError(
                "phrase terms cannot be empty"
            )

        self.terms = terms

    def evaluate(
        self,
        positional_index: PositionalIndex,
    ) -> set[int]:

        if not isinstance(
            positional_index,
            PositionalIndex,
        ):
            raise TypeError(
                "positional_index must be a PositionalIndex"
            )

        # --------------------------------------------------
        # 1. Find candidate documents
        # --------------------------------------------------

        candidate_documents = None

        for term in self.terms:

            documents = (
                positional_index.get_documents(term)
            )

            if documents is None:
                return set()

            documents = set(documents)

            if candidate_documents is None:
                candidate_documents = documents

            else:
                candidate_documents &= documents

            if not candidate_documents:
                return set()

        # --------------------------------------------------
        # 2. Check phrase positions
        # --------------------------------------------------

        matching_documents = set()

        for document_id in candidate_documents:

            if self._matches_in_document(
                positional_index,
                document_id,
            ):
                matching_documents.add(document_id)

        return matching_documents

    def _matches_in_document(
        self,
        positional_index: PositionalIndex,
        document_id: int,
    ) -> bool:

        # --------------------------------------------------
        # Positions of the first term
        # --------------------------------------------------

        first_positions = (
            positional_index.get_positions(
                self.terms[0],
                document_id,
            )
        )

        if first_positions is None:
            return False

        # --------------------------------------------------
        # Try every possible starting position
        # --------------------------------------------------

        for start_position in first_positions:

            matches = True

            for offset, term in enumerate(
                self.terms[1:],
                start=1,
            ):

                expected_position = (
                    start_position + offset
                )

                positions = (
                    positional_index.get_positions(
                        term,
                        document_id,
                    )
                )

                if positions is None:
                    matches = False
                    break

                if expected_position not in positions:
                    matches = False
                    break

            if matches:
                return True

        return False

    def __repr__(self) -> str:

        return (
            f"{self.__class__.__name__}"
            f"({self.terms!r})"
        )

    