from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from retieval.lexical.retriever import RetrievalResult


class FusionStrategy(ABC):
    """
    Abstract interface for combining ranked retrieval results.

    A fusion strategy receives ranked result lists produced by
    independent retrievers and combines them into one ranked list.

    Concrete implementations may use techniques such as:

    - Reciprocal Rank Fusion (RRF)
    - weighted rank fusion
    - other rank-based fusion methods
    """

    @abstractmethod
    def fuse(
        self,
        result_lists: Sequence[Sequence[RetrievalResult]],
        top_k: int = 10,
    ) -> Sequence[RetrievalResult]:
        """
        Combine multiple ranked result lists.

        Parameters
        ----------
        result_lists:
            Ranked result lists produced by independent retrievers.

            Each inner sequence must be ordered from highest
            relevance to lowest relevance.

        top_k:
            Maximum number of fused results to return.

        Returns
        -------
        Sequence[RetrievalResult]
            A single fused ranking.

        Raises
        ------
        TypeError
            If arguments have invalid types.

        ValueError
            If result_lists is empty or top_k is not positive.
        """

        if not isinstance(result_lists, Sequence):
            raise TypeError("result_lists must be a sequence")

        if not result_lists:
            raise ValueError("result_lists must not be empty")

        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise TypeError("top_k must be an integer")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        for results in result_lists:
            if not isinstance(results, Sequence):
                raise TypeError(
                    "each result list must be a sequence"
                )

            for result in results:
                if not isinstance(result, RetrievalResult):
                    raise TypeError(
                        "result lists must contain "
                        "RetrievalResult objects"
                    )

        raise NotImplementedError