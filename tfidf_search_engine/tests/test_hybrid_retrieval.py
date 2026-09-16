import unittest

from analysis.analyzer import Analyzer
from hybrid import HybridRetriever, ReciprocalRankFusion
from index.builder import IndexBuilder
from index.inverted_index import InvertedIndex
from index.vocabulary import Vocabulary
from ranking.bm25 import BM25
from retieval.candidate_generator import CandidateGenerator
from retieval.lexical.bm25_retriever import BM25Retriever
from retieval.lexical.retriever import RetrievalResult


class TestHybridRetrieval(unittest.TestCase):
    def setUp(self) -> None:
        analyzer = Analyzer()
        builder = IndexBuilder(
            analyzer=analyzer,
            vocabulary=Vocabulary(),
            inverted_index=InvertedIndex(),
        )
        builder.add_document(1, "machine learning")
        builder.add_document(2, "machine vision")

        retriever = BM25Retriever(
            candidate_generator=CandidateGenerator(
                builder.inverted_index
            ),
            scorer=BM25(
                builder.forward_index,
                builder.statistics,
            ),
        )
        self.hybrid = HybridRetriever(
            retrievers=[retriever],
            fusion_strategy=ReciprocalRankFusion(),
        )

    def test_hybrid_accepts_real_lexical_retriever(self):
        results = self.hybrid.retrieve(
            "machine learning",
            top_k=1,
        )

        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], RetrievalResult)
        self.assertEqual(results[0].document_id, "1")

    def test_hybrid_top_k_is_forwarded(self):
        results = self.hybrid("machine learning", top_k=1)

        self.assertEqual(len(results), 1)


if __name__ == "__main__":
    unittest.main()
