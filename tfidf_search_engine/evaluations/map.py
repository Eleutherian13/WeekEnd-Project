from __future__ import annotations

from collections.abc import Iterable, Sequence


def average_precision(
    retrieved: Sequence[str],
    relevant: set[str],
) -> float:
    """
    Calculate Average Precision for one query.

    Only relevant documents contribute to the average.
    """

    if not isinstance(retrieved, Sequence):
        raise TypeError("retrieved must be a sequence")

    if not isinstance(relevant, set):
        raise TypeError("relevant must be a set")

    if not relevant:
        return 0.0

    relevant_found = 0
    precision_sum = 0.0
    seen: set[str] = set()

    for rank, document_id in enumerate(retrieved, start=1):

        # Ignore duplicate documents in the ranking.
        if document_id in seen:
            continue

        seen.add(document_id)

        if document_id not in relevant:
            continue

        relevant_found += 1

        precision_at_rank = relevant_found / rank

        precision_sum += precision_at_rank

    return precision_sum / len(relevant)


def mean_average_precision(
    retrieved_results: Iterable[Sequence[str]],
    relevant_results: Iterable[set[str]],
) -> float:
    """
    Calculate Mean Average Precision across queries.
    """

    scores: list[float] = []

    for retrieved, relevant in zip(
        retrieved_results,
        relevant_results,
    ):
        scores.append(
            average_precision(
                retrieved=retrieved,
                relevant=relevant,
            )
        )

    if not scores:
        return 0.0

    return sum(scores) / len(scores)