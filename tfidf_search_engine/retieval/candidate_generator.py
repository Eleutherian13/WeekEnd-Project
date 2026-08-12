
from index.inverted_index import InvertedIndex



class CandidateGenerator:
    """
    Generates candidate document IDs from
    already-processed query terms using
    an inverted index.
    """

    def __init__(
        self,
        inverted_index: InvertedIndex,
    ) -> None:

        if not isinstance(
            inverted_index,
            InvertedIndex,
        ):
            raise TypeError(
                "inverted_index must be an InvertedIndex"
            )

        self.inverted_index = inverted_index

    def generate(
        self,
        terms: list[str],
    ) -> set[int]:

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

        candidate_documents: set[int] = set()

        for term in terms:

            if not self.inverted_index.contain(term):
                continue

            posting_list = (
                self.inverted_index.get_postings(term)
            )

            for posting in posting_list:
                candidate_documents.add(
                    posting.document_id
                )

        return candidate_documents

    def __call__(
        self,
        terms: list[str],
    ) -> set[int]:

        return self.generate(terms)

    def __repr__(self) -> str:

        return (
            f"{self.__class__.__name__}("
            f"inverted_index={self.inverted_index!r}"
            f")"
        )
