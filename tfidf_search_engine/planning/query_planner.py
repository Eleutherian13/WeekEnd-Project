from __future__ import annotations

from dataclasses import dataclass

from .query_analyser import QueryAnalysis, QueryAnalyzer
from .query_classifier import (
    QueryClassification,
    QueryClassifier,
    RetrievalMode,
)


@dataclass(frozen=True, slots=True)
class QueryPlan:
    """
    Execution plan produced for a user query.

    The plan describes which retrieval modes should participate
    and how many candidates should be requested.

    It does not execute retrieval itself.
    """

    query: str
    primary_mode: RetrievalMode
    enabled_modes: tuple[RetrievalMode, ...]
    candidate_k: int
    final_k: int
    max_graph_hops: int
    analysis: QueryAnalysis
    classification: QueryClassification


class QueryPlanner:
    """
    Converts query analysis and classification into a retrieval plan.

    The planner owns routing decisions.
    It does not execute retrievers.
    """

    def __init__(
        self,
        analyzer: QueryAnalyzer | None = None,
        classifier: QueryClassifier | None = None,
        candidate_multiplier: int = 5,
        default_graph_hops: int = 2,
    ) -> None:
        if not isinstance(candidate_multiplier, int):
            raise TypeError(
                "candidate_multiplier must be an integer"
            )

        if isinstance(candidate_multiplier, bool):
            raise TypeError(
                "candidate_multiplier must be an integer"
            )

        if candidate_multiplier <= 0:
            raise ValueError(
                "candidate_multiplier must be greater than 0"
            )

        if not isinstance(default_graph_hops, int):
            raise TypeError(
                "default_graph_hops must be an integer"
            )

        if isinstance(default_graph_hops, bool):
            raise TypeError(
                "default_graph_hops must be an integer"
            )

        if default_graph_hops < 0:
            raise ValueError(
                "default_graph_hops must not be negative"
            )

        self._analyzer = (
            analyzer
            if analyzer is not None
            else QueryAnalyzer()
        )

        self._classifier = (
            classifier
            if classifier is not None
            else QueryClassifier()
        )

        self._candidate_multiplier = candidate_multiplier
        self._default_graph_hops = default_graph_hops

    @property
    def analyzer(self) -> QueryAnalyzer:
        """Return the configured query analyzer."""
        return self._analyzer

    @property
    def classifier(self) -> QueryClassifier:
        """Return the configured query classifier."""
        return self._classifier

    def plan(
        self,
        query: str,
        top_k: int = 10,
    ) -> QueryPlan:
        """
        Analyze, classify, and plan retrieval for a query.
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
            raise ValueError(
                "top_k must be greater than 0"
            )

        analysis = self._analyzer.analyze(query)
        classification = self._classifier.classify(analysis)

        enabled_modes = self._select_modes(
            classification=classification,
            analysis=analysis,
        )

        candidate_k = max(
            top_k,
            top_k * self._candidate_multiplier,
        )

        max_graph_hops = (
            self._default_graph_hops
            if RetrievalMode.GRAPH in enabled_modes
            else 0
        )

        return QueryPlan(
            query=query,
            primary_mode=classification.primary_mode,
            enabled_modes=enabled_modes,
            candidate_k=candidate_k,
            final_k=top_k,
            max_graph_hops=max_graph_hops,
            analysis=analysis,
            classification=classification,
        )

    def _select_modes(
        self,
        classification: QueryClassification,
        analysis: QueryAnalysis,
    ) -> tuple[RetrievalMode, ...]:
        """
        Convert classification signals into concrete retrieval modes.

        This method intentionally favors multiple retrieval modes when
        the query contains multiple strong signals.
        """

        modes: list[RetrievalMode] = []

        lexical_threshold = 1.0
        dense_threshold = 1.0
        graph_threshold = 1.0

        if classification.lexical_score >= lexical_threshold:
            modes.append(RetrievalMode.LEXICAL)

        if classification.dense_score >= dense_threshold:
            modes.append(RetrievalMode.DENSE)

        if classification.graph_score >= graph_threshold:
            modes.append(RetrievalMode.GRAPH)

        # No strong signal: use all available retrieval mechanisms
        # rather than pretending we understand the query better than
        # we actually do.
        if not modes:
            return (
                RetrievalMode.LEXICAL,
                RetrievalMode.DENSE,
                RetrievalMode.GRAPH,
            )

        # If multiple retrieval dimensions are present, use hybrid
        # orchestration.
        if len(modes) > 1:
            return tuple(modes)

        # A question with explanatory language should retain lexical
        # evidence alongside dense retrieval.
        if (
            analysis.has_explanatory_signal
            and RetrievalMode.DENSE in modes
            and RetrievalMode.LEXICAL not in modes
        ):
            return (
                RetrievalMode.LEXICAL,
                RetrievalMode.DENSE,
            )

        return tuple(modes)

    def __call__(
        self,
        query: str,
        top_k: int = 10,
    ) -> QueryPlan:
        """Allow the planner to be called like a function."""
        return self.plan(
            query=query,
            top_k=top_k,
        )