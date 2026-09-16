import unittest

from analysis.analyzer import Analyzer
from index.builder import IndexBuilder
from index.inverted_index import InvertedIndex
from index.vocabulary import Vocabulary
from ranking.bm25 import BM25
from retieval.candidate_generator import CandidateGenerator
from retieval.lexical.bm25_retriever import BM25Retriever
from retieval.lexical.retriever import LexicalRetriever


class TestLexicalRetrieval(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = Analyzer()
        self.builder = IndexBuilder(
            analyzer=self.analyzer,
            vocabulary=Vocabulary(),
            inverted_index=InvertedIndex(),
        )
        self.builder.add_document(1, "machine learning systems")
        self.builder.add_document(2, "machine learning")
        self.builder.add_document(3, "cats and dogs")

        candidate_generator = CandidateGenerator(
            self.builder.inverted_index
        )
        scorer = BM25(
            self.builder.forward_index,
            self.builder.statistics,
        )
        self.retriever = BM25Retriever(
            candidate_generator,
            scorer,
        )

    def test_bm25_retrieves_ranked_candidates(self):
        terms = self.analyzer.analyze("machine learning")

        results = self.retriever.retrieve(terms)

        self.assertEqual([document_id for document_id, _ in results], [2, 1])
        self.assertTrue(all(isinstance(score, float) for _, score in results))
        self.assertGreater(results[0][1], results[1][1])

    def test_unknown_and_empty_queries_return_no_results(self):
        self.assertEqual(self.retriever.retrieve([]), [])
        self.assertEqual(self.retriever.retrieve(["unknown"]), [])

    def test_lexical_retriever_is_callable(self):
        terms = self.analyzer.analyze("machine")

        self.assertEqual(
            self.retriever(terms),
            self.retriever.retrieve(terms),
        )

    def test_abstract_lexical_contract_remains_abstract(self):
        self.assertTrue(issubclass(BM25Retriever, LexicalRetriever))
        self.assertIn("retrieve", LexicalRetriever.__abstractmethods__)


if __name__ == "__main__":
    unittest.main()
