import unittest

from analysis.analyzer import Analyzer
from document.corpus import Corpus
from document.documents import Document
from graph.edge import Edge
from graph.graph import Graph
from graph.node import Node
from hybrid import ReciprocalRankFusion
from index.builder import IndexBuilder
from index.inverted_index import InvertedIndex
from index.vocabulary import Vocabulary
from ranking.bm25 import BM25
from rag import Generator
from query.query_processing import QueryProcessor
from retieval.candidate_generator import CandidateGenerator
from retieval.dense.retriever import DenseRetriever
from retieval.graph.graph_retriever import SimpleGraphRetriever
from retieval.lexical.bm25_retriever import BM25Retriever
from retieval.lexical.retriever import RetrievalResult
from reranking import CrossEncoder, Reranker
from search.search_engine import (
    GenerationFailure,
    RetrievalFailure,
    SearchEngine,
)
from vector.embedding import EmbeddingModel


class DeterministicEmbedding(EmbeddingModel):
    def embed(self, text: str) -> list[float]:
        text = self._validate_text(text).casefold()
        if "machine learning" in text:
            return [1.0, 0.0]
        return [0.0, 1.0]


class DeterministicCrossEncoder(CrossEncoder):
    def score(self, query: str, document: str) -> float:
        self._validate_text(query, "query")
        self._validate_text(document, "document")
        return float(document.casefold().count(query.casefold()))

    def score_batch(self, query: str, documents: list[str]) -> list[float]:
        return [self.score(query, document) for document in documents]

    @staticmethod
    def _validate_text(value: str, name: str) -> None:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string")
        if not value.strip():
            raise ValueError(f"{name} must not be empty")


class DeterministicGenerator(Generator):
    def __init__(self) -> None:
        self.contexts = []

    def generate(self, context):
        self.contexts.append(context)
        return f"Answer grounded in {len(context.evidence)} source(s)."


class TestSearchOrchestrator(unittest.TestCase):
    def setUp(self) -> None:
        self.corpus = Corpus()
        self.corpus.add(Document(1, "machine learning systems"))
        self.corpus.add(Document(2, "machine vision systems"))
        self.corpus.add(Document(3, "cats and dogs"))

        analyzer = Analyzer()
        builder = IndexBuilder(analyzer, Vocabulary(), InvertedIndex())
        for document in self.corpus:
            builder.add_document(document)

        lexical = BM25Retriever(
            CandidateGenerator(builder.inverted_index),
            BM25(builder.forward_index, builder.statistics),
        )
        dense = DenseRetriever.from_documents(
            list(self.corpus),
            DeterministicEmbedding(),
        )

        graph = Graph()
        graph.add_node(
            Node(
                "machine-related",
                "entity",
                {"name": "machine related connected dependency"},
            )
        )
        graph.add_node(Node("1", "document", {"source": "graph"}))
        graph.add_edge(Edge("machine-related", "1", "supports"))
        graph_retriever = SimpleGraphRetriever(graph, max_hops=1)

        self.generator = DeterministicGenerator()
        reranker = Reranker(
            DeterministicCrossEncoder(),
            lambda document_id: self.corpus[int(document_id)].text,
        )
        self.engine = SearchEngine(
            query_processor=QueryProcessor(analyzer),
            lexical_retriever=lexical,
            dense_retriever=dense,
            graph_retriever=graph_retriever,
            reranker=reranker,
            document_text_provider=lambda document_id: self.corpus[
                int(document_id)
            ].text,
            generator=self.generator,
        )

    def test_lexical_retrieval(self):
        results = self.engine.retrieve("machine", top_k=2)

        self.assertEqual(
            [result.document_id for result in results],
            ["1", "2"],
        )

    def test_dense_retrieval(self):
        results = self.engine.retrieve(
            "what is machine learning about",
            top_k=1,
        )

        self.assertEqual([result.document_id for result in results], ["1"])

    def test_graph_retrieval(self):
        results = self.engine.retrieve(
            "related connected dependency",
            top_k=1,
        )

        self.assertEqual([result.document_id for result in results], ["1"])

    def test_hybrid_retrieval_uses_rrf_candidates(self):
        results = self.engine.retrieve('"machine related"', top_k=2)

        self.assertEqual([result.document_id for result in results], ["1", "2"])
        self.assertGreater(results[0].score, results[1].score)

    def test_reranking_and_rag_generation_expose_evidence(self):
        response = self.engine.answer("machine learning", top_k=2)

        self.assertEqual(response.answer, "Answer grounded in 2 source(s).")
        self.assertEqual(
            [result.document_id for result in response.results],
            ["1", "2"],
        )
        self.assertEqual(
            [item.document_id for item in response.evidence],
            ["1", "2"],
        )
        self.assertIsNotNone(response.context)
        self.assertEqual(len(self.generator.contexts), 1)

    def test_top_k_is_respected_after_reranking(self):
        response = self.engine.answer("machine", top_k=1)

        self.assertEqual(len(response.results), 1)
        self.assertEqual(len(response.evidence), 1)

    def test_no_results_does_not_generate(self):
        response = self.engine.answer("quantum", top_k=2)

        self.assertEqual(response.results, ())
        self.assertEqual(response.evidence, ())
        self.assertIsNone(response.answer)
        self.assertEqual(len(self.generator.contexts), 0)

    def test_missing_generator_is_generation_failure(self):
        engine = SearchEngine(
            self.engine.query_processor,
            self.engine.lexical_retriever,
            reranker=self.engine.reranker,
            document_text_provider=self.engine.document_text_provider,
        )

        with self.assertRaises(GenerationFailure):
            engine.answer("machine", top_k=1)

    def test_missing_planned_retriever_is_retrieval_failure(self):
        with self.assertRaises(RetrievalFailure):
            SearchEngine(
                self.engine.query_processor,
                self.engine.lexical_retriever,
            ).retrieve("what is machine learning about")

    def test_invalid_query_is_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.answer("   ")


if __name__ == "__main__":
    unittest.main()