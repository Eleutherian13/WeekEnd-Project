from __future__ import annotations

from collections.abc import Iterable, Sequence


def reciprocal_rank(
    retrieved: Sequence[str],
    relevant: set[str],
) -> float:
    """
    Calculate Reciprocal Rank for a single query.
    """

    if not isinstance(retrieved, Sequence):
        raise TypeError("retrieved must be a sequence")

    if not isinstance(relevant, set):
        raise TypeError("relevant must be a set")

    for rank, document_id in enumerate(retrieved, start=1):
        if document_id in relevant:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    retrieved_results: Iterable[Sequence[str]],
    relevant_results: Iterable[set[str]],
) -> float:
    """
    Calculate Mean Reciprocal Rank across multiple queries.
    """

    reciprocal_ranks: list[float] = []

    for retrieved, relevant in zip(
        retrieved_results,
        relevant_results,
    ):
        reciprocal_ranks.append(
            reciprocal_rank(
                retrieved=retrieved,
                relevant=relevant,
            )
        )

    if not reciprocal_ranks:
        return 0.0

    return sum(reciprocal_ranks) / len(reciprocal_ranks)