from __future__ import annotations

from collections.abc import Sequence

from retieval.lexical.retriever import RetrievalResult

from .fusion import FusionStrategy


class ReciprocalRankFusion(FusionStrategy):
    """
    Reciprocal Rank Fusion (RRF).

    RRF combines ranked result lists using the reciprocal of each
    document's rank.

    For a document d:

        RRF(d) = sum(1 / (k + rank_r(d)))

    where:

        k      = smoothing constant
        rank   = 1-based rank of the document
        r      = a retrieval result list

    RRF deliberately ignores the raw scores produced by individual
    retrievers. This makes it useful when different retrievers produce
    scores on incompatible numerical scales.
    """

    def __init__(self, k: int = 60) -> None:
        if not isinstance(k, int) or isinstance(k, bool):
            raise TypeError("k must be an integer")

        if k <= 0:
            raise ValueError("k must be greater than 0")

        self._k = k

    @property
    def k(self) -> int:
        """Return the RRF smoothing constant."""
        return self._k

    def fuse(
        self,
        result_lists: Sequence[Sequence[RetrievalResult]],
        top_k: int = 10,
    ) -> Sequence[RetrievalResult]:
        """
        Fuse multiple ranked result lists using RRF.
        """

        self._validate_inputs(
            result_lists=result_lists,
            top_k=top_k,
        )

        scores: dict[str, float] = {}
        metadata: dict[str, dict | None] = {}

        for result_list in result_lists:
            seen_in_list: set[str] = set()

            for rank, result in enumerate(result_list, start=1):
                document_id = result.document_id

                # A document should have one rank per retriever.
                # If a malformed result list contains the same document
                # more than once, only its first occurrence counts.
                if document_id in seen_in_list:
                    continue

                seen_in_list.add(document_id)

                contribution = 1.0 / (self._k + rank)

                scores[document_id] = (
                    scores.get(document_id, 0.0)
                    + contribution
                )

                # Preserve the first available metadata.
                if document_id not in metadata:
                    metadata[document_id] = result.metadata

        ranked_documents = sorted(
            scores.items(),
            key=lambda item: (-item[1], item[0]),
        )

        return [
            RetrievalResult(
                document_id=document_id,
                score=score,
                metadata=metadata.get(document_id),
            )
            for document_id, score in ranked_documents[:top_k]
        ]

    @staticmethod
    def _validate_inputs(
        result_lists: Sequence[Sequence[RetrievalResult]],
        top_k: int,
    ) -> None:
        """
        Validate inputs shared by the concrete RRF implementation.
        """

        if not isinstance(result_lists, Sequence):
            raise TypeError(
                "result_lists must be a sequence"
            )

        if not result_lists:
            raise ValueError(
                "result_lists must not be empty"
            )

        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise TypeError(
                "top_k must be an integer"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        for result_list in result_lists:
            if not isinstance(result_list, Sequence):
                raise TypeError(
                    "each result list must be a sequence"
                )

            for result in result_list:
                if not isinstance(result, RetrievalResult):
                    raise TypeError(
                        "result lists must contain "
                        "RetrievalResult objects"
                    )



                