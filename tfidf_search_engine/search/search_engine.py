from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from hybrid import HybridRetriever, ReciprocalRankFusion
from planning import QueryPlanner, RetrievalMode
from query.query import Query
from query.query_processing import QueryProcessor
from rag import ContextBuilder, Evidence, EvidenceBuilder, Generator, RAGContext
from reranking import Reranker
from retieval.dense.retriever import DenseRetriever
from retieval.graph.retriever import GraphRetriever
from retieval.lexical.retriever import LexicalRetriever
from retieval.lexical.retriever import RetrievalResult


class RetrievalFailure(RuntimeError):
    """Raised when planned retrieval or reranking cannot complete."""


class GenerationFailure(RuntimeError):
    """Raised when answer generation cannot complete."""


@dataclass(frozen=True, slots=True)
class RAGResponse:
    """Complete retrieval and generation output."""

    query: str
    results: tuple[RetrievalResult, ...]
    evidence: tuple[Evidence, ...]
    context: RAGContext | None
    answer: str | None


class SearchEngine:
    """Lexical search facade over the existing retriever contract."""

    def __init__(
        self,
        query_processor: QueryProcessor,
        lexical_retriever: LexicalRetriever,
        dense_retriever: DenseRetriever | None = None,
        graph_retriever: GraphRetriever | None = None,
        planner: QueryPlanner | None = None,
        reranker: Reranker | None = None,
        document_text_provider: Callable[[str], str] | None = None,
        context_builder: ContextBuilder | None = None,
        generator: Generator | None = None,
    ) -> None:
        if not isinstance(query_processor, QueryProcessor):
            raise TypeError(
                "query_processor must be a QueryProcessor"
            )

        if not isinstance(lexical_retriever, LexicalRetriever):
            raise TypeError(
                "lexical_retriever must be a LexicalRetriever"
            )

        if dense_retriever is not None and not isinstance(
            dense_retriever,
            DenseRetriever,
        ):
            raise TypeError(
                "dense_retriever must be a DenseRetriever or None"
            )

        if graph_retriever is not None and not isinstance(
            graph_retriever,
            GraphRetriever,
        ):
            raise TypeError(
                "graph_retriever must be a GraphRetriever or None"
            )

        if planner is not None and not isinstance(planner, QueryPlanner):
            raise TypeError("planner must be a QueryPlanner or None")

        if reranker is not None and not isinstance(reranker, Reranker):
            raise TypeError("reranker must be a Reranker or None")

        if document_text_provider is not None and not callable(
            document_text_provider
        ):
            raise TypeError(
                "document_text_provider must be callable or None"
            )

        if context_builder is not None and not isinstance(
            context_builder,
            ContextBuilder,
        ):
            raise TypeError(
                "context_builder must be a ContextBuilder or None"
            )

        if generator is not None and not isinstance(generator, Generator):
            raise TypeError("generator must be a Generator or None")

        self.query_processor = query_processor
        self.lexical_retriever = lexical_retriever
        self.dense_retriever = dense_retriever
        self.graph_retriever = graph_retriever
        self.planner = planner or QueryPlanner()
        self.reranker = reranker
        self.document_text_provider = document_text_provider
        self.context_builder = context_builder or ContextBuilder()
        self.generator = generator
        self._hybrid = HybridRetriever(
            retrievers=self._configured_retrievers(),
            fusion_strategy=ReciprocalRankFusion(),
            query_processor=query_processor,
            planner=self.planner,
        )

    def _configured_retrievers(self) -> tuple[object, ...]:
        retrievers: list[object] = [self.lexical_retriever]
        if self.dense_retriever is not None:
            retrievers.append(self.dense_retriever)
        if self.graph_retriever is not None:
            retrievers.append(self.graph_retriever)
        return tuple(retrievers)

    def retrieve(
        self,
        text: str,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Retrieve planner-selected candidates without generation."""

        self._validate_query_arguments(text, top_k)
        plan = self.planner.plan(text, top_k=top_k)
        self._validate_configured_modes(plan.enabled_modes)

        try:
            results = self._hybrid.retrieve(text, top_k=top_k)
        except (TypeError, ValueError):
            raise
        except Exception as error:
            raise RetrievalFailure(
                "planned retrieval failed"
            ) from error

        return list(results)

    def answer(
        self,
        text: str,
        top_k: int = 10,
    ) -> RAGResponse:
        """Run retrieval, reranking, evidence building, and generation."""

        self._validate_query_arguments(text, top_k)
        results = self.retrieve(text, top_k=top_k)

        if not results:
            return RAGResponse(
                query=text,
                results=(),
                evidence=(),
                context=self.context_builder.build(text, []),
                answer=None,
            )

        if self.reranker is None:
            raise RetrievalFailure(
                "reranking is required for the complete RAG pipeline"
            )

        if self.document_text_provider is None:
            raise RetrievalFailure(
                "document_text_provider is required for the complete RAG pipeline"
            )

        try:
            reranked = self.reranker.rerank(
                query=text,
                candidates=results,
                top_k=top_k,
            )
            evidence = EvidenceBuilder(
                self.document_text_provider
            ).build(reranked)
            context = self.context_builder.build(
                query=text,
                evidence=list(evidence),
            )
        except Exception as error:
            raise RetrievalFailure(
                "reranking or evidence construction failed"
            ) from error

        if self.generator is None:
            raise GenerationFailure(
                "a generator is required to produce an answer"
            )

        try:
            generated_answer = self.generator.generate(context)
        except Exception as error:
            raise GenerationFailure(
                "answer generation failed"
            ) from error

        return RAGResponse(
            query=text,
            results=tuple(reranked),
            evidence=tuple(evidence),
            context=context,
            answer=generated_answer,
        )

    def _validate_configured_modes(
        self,
        enabled_modes: Sequence[RetrievalMode],
    ) -> None:
        configured_modes = {RetrievalMode.LEXICAL}
        if self.dense_retriever is not None:
            configured_modes.add(RetrievalMode.DENSE)
        if self.graph_retriever is not None:
            configured_modes.add(RetrievalMode.GRAPH)

        missing_modes = set(enabled_modes) - configured_modes
        if missing_modes:
            names = ", ".join(
                sorted(mode.value for mode in missing_modes)
            )
            raise RetrievalFailure(
                f"planner selected unconfigured retrieval mode(s): {names}"
            )

    @staticmethod
    def _validate_query_arguments(text: str, top_k: int) -> None:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text.strip():
            raise ValueError("text must not be empty")

        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise TypeError("top_k must be an integer")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

    def search(
        self,
        text: str,
        mode: str = "OR",
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if not isinstance(mode, str):
            raise TypeError("mode must be a string")

        mode = mode.upper()
        if mode not in {"AND", "OR"}:
            raise ValueError(f"unsupported retrieval mode: {mode}")

        query = Query(text)
        terms = self.query_processor.process(query)
        results = self.lexical_retriever.retrieve(terms)

        if mode == "AND" and terms:
            candidate_generator = (
                self.lexical_retriever.candidate_generator
            )
            matching_documents = set.intersection(
                *(
                    candidate_generator.generate([term])
                    for term in terms
                )
            )
            results = [
                result
                for result in results
                if result[0] in matching_documents
            ]

        return results[:top_k]

    def __call__(
        self,
        text: str,
        mode: str = "OR",
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        return self.search(
            text=text,
            top_k=top_k,
            mode=mode,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"







