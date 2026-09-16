from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from retieval.lexical.retriever import RetrievalResult


class GraphRetriever(ABC):
    """
    Abstract interface for graph-based retrieval.

    A graph retriever uses graph structure to discover documents,
    chunks, entities, or other retrievable resources connected
    through graph relationships.

    Concrete implementations are responsible for deciding:

    - how query concepts map to graph nodes
    - which nodes are used as traversal starting points
    - how the graph is traversed
    - how graph evidence is converted into retrieval scores
    - how final document candidates are ranked
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> Sequence[RetrievalResult]:
        """
        Retrieve the top-k documents relevant to the query.

        Parameters
        ----------
        query:
            User's natural-language query.

        top_k:
            Maximum number of results to return.

        Returns
        -------
        Sequence[RetrievalResult]
            Retrieved documents ordered from most relevant
            to least relevant.

        Raises
        ------
        TypeError
            If query is not a string or top_k is not an integer.

        ValueError
            If query is empty or top_k is not positive.
        """

        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if not query.strip():
            raise ValueError("query must not be empty")

        if not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")

        if isinstance(top_k, bool):
            raise TypeError("top_k must be an integer")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        raise NotImplementedError

    def __call__(
        self,
        query: str,
        top_k: int = 10,
    ) -> Sequence[RetrievalResult]:
        """
        Allow the retriever to be called like a function.
        """
        return self.retrieve(
            query=query,
            top_k=top_k,
        )