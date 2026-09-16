from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final


_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"\b[\w-]+\b")
_QUOTED_PATTERN: Final[re.Pattern[str]] = re.compile(r'"([^"]+)"')


@dataclass(frozen=True, slots=True)
class QueryAnalysis:
    """
    Structured representation of observable query characteristics.

    This class does not decide which retrieval strategy should be used.
    It only describes the query.
    """

    original_query: str
    normalized_query: str
    terms: tuple[str, ...]
    quoted_phrases: tuple[str, ...]
    term_count: int
    has_exact_phrase: bool
    has_question_form: bool
    has_comparison_signal: bool
    has_relationship_signal: bool
    has_identifier_signal: bool
    has_explanatory_signal: bool


class QueryAnalyzer:
    """
    Performs deterministic query analysis.

    The analyzer extracts surface-level signals from a query.
    It does not perform semantic classification or retrieval.
    """

    _QUESTION_WORDS: Final[frozenset[str]] = frozenset(
        {
            "what",
            "why",
            "how",
            "when",
            "where",
            "which",
            "who",
        }
    )

    _COMPARISON_WORDS: Final[frozenset[str]] = frozenset(
        {
            "compare",
            "comparison",
            "versus",
            "vs",
            "difference",
            "different",
            "similar",
            "similarity",
            "better",
            "contrast",
        }
    )

    _RELATIONSHIP_WORDS: Final[frozenset[str]] = frozenset(
        {
            "related",
            "relationship",
            "connected",
            "connection",
            "depends",
            "dependency",
            "dependencies",
            "linked",
            "links",
            "associated",
            "between",
            "upstream",
            "downstream",
        }
    )

    _EXPLANATORY_WORDS: Final[frozenset[str]] = frozenset(
        {
            "explain",
            "explanation",
            "describe",
            "description",
            "overview",
            "summarize",
            "summary",
            "reason",
            "reasons",
        }
    )

    _IDENTIFIER_PATTERN: Final[re.Pattern[str]] = re.compile(
        r"""
        (
            \bCVE-\d{4}-\d{4,}\b
            |
            \b[A-Z]{2,}[._-]\d+[A-Z0-9._-]*\b
            |
            \b\d{4}[-/]\d{2}[-/]\d{2}\b
        )
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    def analyze(self, query: str) -> QueryAnalysis:
        """
        Analyze a natural-language query.

        Raises
        ------
        TypeError
            If query is not a string.

        ValueError
            If query is empty or whitespace-only.
        """

        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if not query.strip():
            raise ValueError("query must not be empty")

        original_query = query
        normalized_query = " ".join(query.lower().split())

        terms = tuple(
            match.group(0).lower()
            for match in _TOKEN_PATTERN.finditer(normalized_query)
        )

        quoted_phrases = tuple(
            phrase.strip()
            for phrase in _QUOTED_PATTERN.findall(query)
            if phrase.strip()
        )

        term_set = set(terms)

        return QueryAnalysis(
            original_query=original_query,
            normalized_query=normalized_query,
            terms=terms,
            quoted_phrases=quoted_phrases,
            term_count=len(terms),
            has_exact_phrase=bool(quoted_phrases),
            has_question_form=(
                "?" in query
                or bool(term_set & self._QUESTION_WORDS)
            ),
            has_comparison_signal=bool(
                term_set & self._COMPARISON_WORDS
            ),
            has_relationship_signal=bool(
                term_set & self._RELATIONSHIP_WORDS
            ),
            has_identifier_signal=bool(
                self._IDENTIFIER_PATTERN.search(query)
            ),
            has_explanatory_signal=bool(
                term_set & self._EXPLANATORY_WORDS
            ),
        )

    def __call__(self, query: str) -> QueryAnalysis:
        """Allow the analyzer to be called like a function."""
        return self.analyze(query)