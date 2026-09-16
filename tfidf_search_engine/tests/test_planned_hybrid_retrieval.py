import unittest

from document.documents import Document
from analysis.analyzer import Analyzer
from graph.edge import Edge
from graph.graph import Graph
from graph.node import Node
from hybrid import HybridRetriever, ReciprocalRankFusion
from index.builder import IndexBuilder
from index.inverted_index import InvertedIndex
from index.vocabulary import Vocabulary
from planning import QueryPlanner, RetrievalMode
from ranking.bm25 import BM25
from retieval.candidate_generator import CandidateGenerator
from retieval.dense.retriever import DenseRetriever
from retieval.graph.graph_retriever import SimpleGraphRetriever
from retieval.lexical.bm25_retriever import BM25Retriever
from retieval.lexical.retriever import RetrievalResult
from vector.embedding import EmbeddingModel


class DeterministicEmbedding(EmbeddingModel):
    def embed(self, text: str) -> list[float]:
        text = self._validate_text(text).casefold()
        if "machine learning" in text:
            return [1.0, 0.0]
        return [0.0, 1.0]


class TestPlannedHybridRetrieval(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = [
            Document(1, "machine related learning"),
            Document(2, "machine vision"),
            Document(3, "cats and dogs"),
        ]

        analyzer_builder = IndexBuilder(
            analyzer=Analyzer(),
            vocabulary=Vocabulary(),
            inverted_index=InvertedIndex(),
        )
        for document in self.documents:
            analyzer_builder.add_document(document)

        self.lexical = BM25Retriever(
            candidate_generator=CandidateGenerator(
                analyzer_builder.inverted_index,
            ),
            scorer=BM25(
                analyzer_builder.forward_index,
                analyzer_builder.statistics,
            ),
        )
        self.dense = DenseRetriever.from_documents(
            self.documents,
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
        self.graph = SimpleGraphRetriever(graph, max_hops=1)

        self.hybrid = HybridRetriever(
            retrievers=[self.lexical, self.dense, self.graph],
            fusion_strategy=ReciprocalRankFusion(k=10),
        )

    def test_planner_selects_lexical_only(self):
        plan = self.hybrid.planner.plan("machine", top_k=2)

        self.assertEqual(plan.enabled_modes, (RetrievalMode.LEXICAL,))

    def test_planner_selects_dense_only(self):
        plan = self.hybrid.planner.plan(
            "what is machine learning about",
            top_k=2,
        )

        self.assertEqual(plan.enabled_modes, (RetrievalMode.DENSE,))

    def test_planner_selects_graph_only(self):
        plan = self.hybrid.planner.plan(
            "related connected dependency",
            top_k=2,
        )

        self.assertEqual(plan.enabled_modes, (RetrievalMode.GRAPH,))

    def test_hybrid_classification_enables_multiple_modes(self):
        plan = QueryPlanner().plan('"machine related"', top_k=2)

        self.assertEqual(plan.primary_mode, RetrievalMode.HYBRID)
        self.assertEqual(
            plan.enabled_modes,
            (RetrievalMode.LEXICAL, RetrievalMode.GRAPH),
        )

    def test_real_lexical_and_graph_results_are_fused(self):
        results = self.hybrid.retrieve('"machine related"', top_k=2)

        self.assertEqual(
            [result.document_id for result in results],
            ["1", "2"],
        )
        self.assertIsInstance(results[0], RetrievalResult)
        self.assertGreater(results[0].score, results[1].score)

    def test_dense_only_execution_returns_real_dense_results(self):
        results = self.hybrid.retrieve(
            "what is machine learning about",
            top_k=1,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].document_id, "1")
        self.assertIsInstance(results[0], RetrievalResult)

    def test_top_k_applies_after_fusion(self):
        results = self.hybrid.retrieve("machine", top_k=1)

        self.assertEqual(len(results), 1)


if __name__ == "__main__":
    unittest.main()