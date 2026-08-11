from index.vocabulary import Vocabulary
from index.inverted_index import InvertedIndex


class SynonymQuery:
    """
    Retrieves documents containing the query term
    or any of its configured synonyms.
    """

    def __init__(
        self,
        term: str,
        synonyms: dict[str, list[str]],
    ) -> None:

        if not isinstance(term, str):
            raise TypeError(
                "term must be a string"
            )

        if not term:
            raise ValueError(
                "term cannot be empty"
            )

        if not isinstance(synonyms, dict):
            raise TypeError(
                "synonyms must be a dictionary"
            )

        for key, values in synonyms.items():

            if not isinstance(key, str):
                raise TypeError(
                    "synonym keys must be strings"
                )

            if not isinstance(values, list):
                raise TypeError(
                    "synonyms must contain lists"
                )

            if not all(
                isinstance(value, str)
                for value in values
            ):
                raise TypeError(
                    "synonym values must be strings"
                )

        self.term = term
        self.synonyms = synonyms

    def matching_terms(self) -> list[str]:

        terms = [self.term]

        for synonym in self.synonyms.get(
            self.term,
            [],
        ):

            if synonym not in terms:
                terms.append(synonym)

        return terms

    def evaluate(
        self,
        inverted_index: InvertedIndex,
    ) -> set[int]:

        if not isinstance(
            inverted_index,
            InvertedIndex,
        ):
            raise TypeError(
                "inverted_index must be an InvertedIndex"
            )

        documents = set()

        for term in self.matching_terms():

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
            f"{self.__class__.__name__}"
            f"({self.term!r})"
        )