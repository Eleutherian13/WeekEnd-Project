from __future__ import annotations

from dataclasses import dataclass

from .map import average_precision
from .mrr import reciprocal_rank
from .ndgc import ndcg
from .prevision import precision, precision_at_k
from .recall import recall, recall_at_k


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """
    Contains evaluation metrics for a single query.
    """

    precision: float
    recall: float
    reciprocal_rank: float
    average_precision: float
    ndcg: float
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float


class Evaluator:
    """
    Coordinates retrieval evaluation.

    The Evaluator knows about evaluation metrics,
    but does not know how retrieval itself works.
    """

    def evaluate(
        self,
        retrieved: list[str],
        relevant: set[str],
        relevance_scores: dict[str, float] | None = None,
        k: int = 10,
    ) -> EvaluationResult:
        """
        Evaluate a single ranked retrieval result.
        """

        if not isinstance(retrieved, list):
            raise TypeError("retrieved must be a list")

        if not isinstance(relevant, set):
            raise TypeError("relevant must be a set")

        if relevance_scores is not None:
            if not isinstance(relevance_scores, dict):
                raise TypeError(
                    "relevance_scores must be a dictionary or None"
                )

        if not isinstance(k, int):
            raise TypeError("k must be an integer")

        if k <= 0:
            raise ValueError("k must be greater than 0")

        binary_retrieved_relevance = [
            1.0 if document_id in relevant else 0.0
            for document_id in retrieved
        ]

        if relevance_scores is None:
            ranked_relevance = binary_retrieved_relevance
        else:
            ranked_relevance = [
                relevance_scores.get(document_id, 0.0)
                for document_id in retrieved
            ]

        return EvaluationResult(
            precision=precision(
                retrieved=retrieved,
                relevant=relevant,
            ),
            recall=recall(
                retrieved=retrieved,
                relevant=relevant,
            ),
            reciprocal_rank=reciprocal_rank(
                retrieved=retrieved,
                relevant=relevant,
            ),
            average_precision=average_precision(
                retrieved=retrieved,
                relevant=relevant,
            ),
            ndcg=ndcg(
                relevance_scores=ranked_relevance,
            ),
            precision_at_k=precision_at_k(
                retrieved=retrieved,
                relevant=relevant,
                k=k,
            ),
            recall_at_k=recall_at_k(
                retrieved=retrieved,
                relevant=relevant,
                k=k,
            ),
            ndcg_at_k=ndcg(
                relevance_scores=ranked_relevance,
                k=k,
            ),
        )