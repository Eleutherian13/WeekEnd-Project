from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .query_analyser import QueryAnalysis


class RetrievalMode(str, Enum):
    """
    High-level retrieval modes supported by the engine.
    """

    LEXICAL = "lexical"
    DENSE = "dense"
    GRAPH = "graph"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class QueryClassification:
    """
    Classification result produced from QueryAnalysis.
    """

    primary_mode: RetrievalMode
    lexical_score: float
    dense_score: float
    graph_score: float
    confidence: float


class QueryClassifier:
    """
    Deterministic baseline query classifier.

    The classifier uses observable query signals rather than an
    external machine-learning model.

    These scores are routing signals, not retrieval relevance scores.
    """

    def classify(
        self,
        analysis: QueryAnalysis,
    ) -> QueryClassification:
        """
        Classify a previously analyzed query.
        """

        if not isinstance(analysis, QueryAnalysis):
            raise TypeError(
                "analysis must be a QueryAnalysis"
            )

        lexical = 0.0
        dense = 0.0
        graph = 0.0

        # Exact phrases and identifiers strongly benefit from
        # lexical matching.
        if analysis.has_exact_phrase:
            lexical += 3.0

        if analysis.has_identifier_signal:
            lexical += 4.0

        # Relationship-oriented language is a strong graph signal.
        if analysis.has_relationship_signal:
            graph += 4.0

        # Comparisons often benefit from both semantic and graph
        # retrieval.
        if analysis.has_comparison_signal:
            dense += 2.0
            graph += 2.0

        # Explanatory questions generally benefit from semantic
        # retrieval in addition to lexical evidence.
        if analysis.has_explanatory_signal:
            dense += 2.0

        if analysis.has_question_form:
            dense += 1.0

        # Very short queries are usually better served by exact
        # lexical matching as a baseline.
        if analysis.term_count <= 2:
            lexical += 1.0

        # Longer natural-language queries contain more semantic
        # context.
        if analysis.term_count >= 6:
            dense += 1.0

        scores = {
            RetrievalMode.LEXICAL: lexical,
            RetrievalMode.DENSE: dense,
            RetrievalMode.GRAPH: graph,
        }

        highest = max(scores.values())

        if highest == 0.0:
            primary_mode = RetrievalMode.HYBRID
            confidence = 0.0
        else:
            winners = [
                mode
                for mode, score in scores.items()
                if score == highest
            ]

            if len(winners) > 1:
                primary_mode = RetrievalMode.HYBRID
            else:
                primary_mode = winners[0]

            total = lexical + dense + graph

            if total == 0.0:
                confidence = 0.0
            else:
                confidence = highest / total

                if len(winners) > 1:
                    confidence = min(confidence, 0.5)

        return QueryClassification(
            primary_mode=primary_mode,
            lexical_score=lexical,
            dense_score=dense,
            graph_score=graph,
            confidence=confidence,
        )

    def __call__(
        self,
        analysis: QueryAnalysis,
    ) -> QueryClassification:
        """Allow the classifier to be called like a function."""
        return self.classify(analysis)