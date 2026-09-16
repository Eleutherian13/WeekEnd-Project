from __future__ import annotations

from collections.abc import Sequence


def precision(
    retrieved: Sequence[str],
    relevant: set[str],
) -> float:
    """
    Calculate precision.

    Precision measures the fraction of retrieved documents
    that are relevant.

    Formula:

        precision = relevant_retrieved / retrieved

    Parameters
    ----------
    retrieved:
        Ranked or unranked document IDs returned by the system.

    relevant:
        Set of relevant document IDs.

    Returns
    -------
    float
        Precision in the range [0, 1].
    """

    if not isinstance(retrieved, Sequence):
        raise TypeError("retrieved must be a sequence")

    if not isinstance(relevant, set):
        raise TypeError("relevant must be a set")

    if not retrieved:
        return 0.0

    relevant_retrieved = sum(
        document_id in relevant
        for document_id in retrieved
    )

    return relevant_retrieved / len(retrieved)


def precision_at_k(
    retrieved: Sequence[str],
    relevant: set[str],
    k: int,
) -> float:
    """
    Calculate Precision@K.
    """

    if not isinstance(k, int):
        raise TypeError("k must be an integer")

    if k <= 0:
        raise ValueError("k must be greater than 0")

    return precision(
        retrieved=retrieved[:k],
        relevant=relevant,
    )