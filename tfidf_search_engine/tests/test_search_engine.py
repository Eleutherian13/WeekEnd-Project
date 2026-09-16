import unittest

from analysis.analyzer import Analyzer
from index.builder import IndexBuilder
from index.inverted_index import InvertedIndex
from index.vocabulary import Vocabulary
from query.query_processing import QueryProcessor
from ranking.bm25 import BM25
from retieval.candidate_generator import CandidateGenerator
from retieval.lexical.bm25_retriever import BM25Retriever
from search.search_engine import SearchEngine


class TestSearchEngine(unittest.TestCase):
    def setUp(self) -> None:
        analyzer = Analyzer()
        builder = IndexBuilder(
            analyzer=analyzer,
            vocabulary=Vocabulary(),
            inverted_index=InvertedIndex(),
        )
        builder.add_document(1, "machine learning")
        builder.add_document(2, "machine vision")
        builder.add_document(3, "cats and dogs")

        retriever = BM25Retriever(
            candidate_generator=CandidateGenerator(
                builder.inverted_index
            ),
            scorer=BM25(
                builder.forward_index,
                builder.statistics,
            ),
        )

        self.engine = SearchEngine(
            query_processor=QueryProcessor(analyzer),
            lexical_retriever=retriever,
        )

    def test_valid_query_returns_ranked_lexical_results(self):
        results = self.engine.search("machine learning")

        self.assertEqual([result[0] for result in results], [1, 2])
        self.assertGreaterEqual(results[0][1], results[1][1])

    def test_and_mode_returns_intersection(self):
        self.assertEqual(
            [result[0] for result in self.engine.search(
                "machine learning",
                mode="AND",
            )],
            [1],
        )

    def test_top_k_limits_results(self):
        self.assertEqual(
            len(self.engine.search("machine learning", top_k=1)),
            1,
        )

    def test_top_k_must_be_positive_integer(self):
        with self.assertRaises(TypeError):
            self.engine.search("machine", top_k=True)

        with self.assertRaises(ValueError):
            self.engine.search("machine", top_k=0)

    def test_empty_query_is_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.search("   ")

    def test_empty_result_query_returns_empty_list(self):
        self.assertEqual(
            self.engine.search("quantum entanglement"),
            [],
        )

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.search("machine", mode="UNKNOWN")

    def test_invalid_mode_type_is_rejected(self):
        with self.assertRaises(TypeError):
            self.engine.search("machine", mode=1)

    def test_callable_interface_uses_actual_retriever(self):
        self.assertEqual(
            self.engine("machine", top_k=1),
            self.engine.search("machine", top_k=1),
        )


if __name__ == "__main__":
    unittest.main()
