# `retrieval/candidate_generator.py`

from __future__ import annotations

from collections.abc import Iterable

from index.inverted_index import InvertedIndex


class CandidateGenerator:
    """
    Generate candidate document IDs from processed query terms.

    Candidate generation answers:

        "Which documents should be considered for ranking?"

    It does not calculate relevance scores.

    Parameters
    ----------
    inverted_index:
        Existing inverted index used to retrieve posting lists for terms.
    """

    def __init__(self, inverted_index: InvertedIndex) -> None:
        if not isinstance(inverted_index, InvertedIndex):
            raise TypeError(
                "inverted_index must be an instance of InvertedIndex."
            )

        self.inverted_index = inverted_index

    def generate(self, query_terms: Iterable[str]) -> set[int]:
        """
        Generate candidate document IDs for the supplied query terms.

        Candidate generation uses OR semantics:

            candidates =
                postings(term_1)
                UNION
                postings(term_2)
                UNION
                ...

        Parameters
        ----------
        query_terms:
            Already processed query terms.

        Returns
        -------
        set[int]
            Unique candidate document IDs.
        """

        if query_terms is None:
            raise TypeError("query_terms cannot be None.")

        candidates: set[int] = set()

        for term in query_terms:
            if not isinstance(term, str):
                raise TypeError("Every query term must be a string.")

            term = term.strip()

            if not term:
                continue

            if not self.inverted_index.contain(term):
                continue

            posting_list = self.inverted_index.get_postings(term)

            for posting in posting_list:
                candidates.add(posting.document_id)

        return candidates

    def __call__(self, query_terms: Iterable[str]) -> set[int]:
        """
        Callable interface for candidate generation.
        """
        return self.generate(query_terms)

    def __repr__(self) -> str:
        return (
            f"CandidateGenerator("
            f"inverted_index={self.inverted_index!r})"
        )