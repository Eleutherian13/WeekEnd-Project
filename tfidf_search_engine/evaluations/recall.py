from __future__ import annotations

from collections.abc import Sequence


def recall(
    retrieved: Sequence[str],
    relevant: set[str],
) -> float:
    """
    Calculate recall.

    Formula:

        recall = relevant_retrieved / total_relevant
    """

    if not isinstance(retrieved, Sequence):
        raise TypeError("retrieved must be a sequence")

    if not isinstance(relevant, set):
        raise TypeError("relevant must be a set")

    if not relevant:
        return 0.0

    retrieved_relevant = sum(
        document_id in relevant
        for document_id in retrieved
    )

    return retrieved_relevant / len(relevant)


def recall_at_k(
    retrieved: Sequence[str],
    relevant: set[str],
    k: int,
) -> float:
    """
    Calculate Recall@K.
    """

    if not isinstance(k, int):
        raise TypeError("k must be an integer")

    if k <= 0:
        raise ValueError("k must be greater than 0")

    return recall(
        retrieved=retrieved[:k],
        relevant=relevant,
    )