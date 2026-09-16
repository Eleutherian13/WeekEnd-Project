from __future__ import annotations

import math
from collections.abc import Sequence


def dcg(
    relevance_scores: Sequence[float],
    k: int | None = None,
) -> float:
    """
    Calculate Discounted Cumulative Gain.

    Higher relevance contributes more, but relevance appearing
    later in the ranking contributes less.
    """

    if k is None:
        k = len(relevance_scores)

    if not isinstance(k, int):
        raise TypeError("k must be an integer or None")

    if k <= 0:
        raise ValueError("k must be greater than 0")

    total = 0.0

    for rank, relevance in enumerate(
        relevance_scores[:k],
        start=1,
    ):
        total += (
            (2**relevance - 1)
            / math.log2(rank + 1)
        )

    return total


def ndcg(
    relevance_scores: Sequence[float],
    k: int | None = None,
) -> float:
    """
    Calculate normalized discounted cumulative gain.
    """

    if k is None:
        k = len(relevance_scores)

    if not relevance_scores:
        return 0.0

    actual_dcg = dcg(
        relevance_scores=relevance_scores,
        k=k,
    )

    ideal_scores = sorted(
        relevance_scores,
        reverse=True,
    )

    ideal_dcg = dcg(
        relevance_scores=ideal_scores,
        k=k,
    )

    if ideal_dcg == 0.0:
        return 0.0

    return actual_dcg / ideal_dcg