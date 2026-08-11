from abc import ABC, abstractmethod

from index.inverted_index import InvertedIndex


class BooleanQuery(ABC):
    """
    Base class for Boolean query expressions.

    Every Boolean query can be evaluated against
    an inverted index and returns a set of document IDs.
    """

    @abstractmethod
    def evaluate(
        self,
        inverted_index: InvertedIndex,
    ) -> set[int]:
        """
        Evaluate the query against the inverted index.
        """
        raise NotImplementedError


class TermQuery(BooleanQuery):
    """
    Represents a single search term.

    Example:
        machine
    """

    def __init__(self, term: str) -> None:

        if not isinstance(term, str):
            raise TypeError("term must be a string")

        if not term:
            raise ValueError("term cannot be empty")

        self.term = term

    def evaluate(
        self,
        inverted_index: InvertedIndex,
    ) -> set[int]:

        posting_list = inverted_index.get(self.term)

        if posting_list is None:
            return set()

        return {
            posting.document_id
            for posting in posting_list
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"({self.term!r})"
        )


class AndQuery(BooleanQuery):
    """
    Represents a Boolean AND expression.

    Example:
        machine AND learning
    """

    def __init__(
        self,
        left: BooleanQuery,
        right: BooleanQuery,
    ) -> None:

        if not isinstance(left, BooleanQuery):
            raise TypeError(
                "left must be a BooleanQuery"
            )

        if not isinstance(right, BooleanQuery):
            raise TypeError(
                "right must be a BooleanQuery"
            )

        self.left = left
        self.right = right

    def evaluate(
        self,
        inverted_index: InvertedIndex,
    ) -> set[int]:

        left_documents = self.left.evaluate(
            inverted_index
        )

        right_documents = self.right.evaluate(
            inverted_index
        )

        return left_documents & right_documents

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.left!r}, "
            f"{self.right!r}"
            f")"
        )


class OrQuery(BooleanQuery):
    """
    Represents a Boolean OR expression.

    Example:
        machine OR learning
    """

    def __init__(
        self,
        left: BooleanQuery,
        right: BooleanQuery,
    ) -> None:

        if not isinstance(left, BooleanQuery):
            raise TypeError(
                "left must be a BooleanQuery"
            )

        if not isinstance(right, BooleanQuery):
            raise TypeError(
                "right must be a BooleanQuery"
            )

        self.left = left
        self.right = right

    def evaluate(
        self,
        inverted_index: InvertedIndex,
    ) -> set[int]:

        left_documents = self.left.evaluate(
            inverted_index
        )

        right_documents = self.right.evaluate(
            inverted_index
        )

        return left_documents | right_documents

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.left!r}, "
            f"{self.right!r}"
            f")"
        )


class NotQuery(BooleanQuery):
    """
    Represents a Boolean NOT expression.

    Example:
        NOT machine
    """

    def __init__(
        self,
        query: BooleanQuery,
        all_documents: set[int],
    ) -> None:

        if not isinstance(query, BooleanQuery):
            raise TypeError(
                "query must be a BooleanQuery"
            )

        if not isinstance(all_documents, set):
            raise TypeError(
                "all_documents must be a set"
            )

        self.query = query
        self.all_documents = all_documents

    def evaluate(
        self,
        inverted_index: InvertedIndex,
    ) -> set[int]:

        matching_documents = self.query.evaluate(
            inverted_index
        )

        return self.all_documents - matching_documents

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.query!r}"
            f")"
        )