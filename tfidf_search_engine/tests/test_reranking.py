import unittest
from unittest.mock import patch

from analysis.analyzer import Analyzer
from document.corpus import Corpus
from document.documents import Document
from index.builder import IndexBuilder
from index.inverted_index import InvertedIndex
from index.vocabulary import Vocabulary
from ranking.bm25 import BM25
from retieval.candidate_generator import CandidateGenerator
from retieval.lexical.bm25_retriever import BM25Retriever
from retieval.lexical.retriever import RetrievalResult
from reranking import (
    CrossEncoder,
    Reranker,
    SentenceTransformerCrossEncoder,
)


class DeterministicCrossEncoder(CrossEncoder):
    def score(self, query: str, document: str) -> float:
        self._validate(query, document)
        return float(document.casefold().count(query.casefold()))

    def score_batch(self, query: str, documents: list[str]) -> list[float]:
        self._validate_query(query)
        if not isinstance(documents, list):
            raise TypeError("documents must be a sequence")
        return [self.score(query, document) for document in documents]

    @staticmethod
    def _validate_query(query: str) -> None:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query must not be empty")

    @classmethod
    def _validate(cls, query: str, document: str) -> None:
        cls._validate_query(query)
        if not isinstance(document, str):
            raise TypeError("document must be a string")
        if not document.strip():
            raise ValueError("document must not be empty")


class TestReranking(unittest.TestCase):
    def setUp(self) -> None:
        self.corpus = Corpus()
        self.corpus.add(Document(1, "machine learning guide"))
        self.corpus.add(Document(2, "machine learning machine"))
        self.corpus.add(Document(3, "cats and dogs"))
        self.corpus.add(Document(4, "machine learning guide"))
        self.reranker = Reranker(
            cross_encoder=DeterministicCrossEncoder(),
            document_text_provider=lambda document_id: self.corpus[
                int(document_id)
            ].text,
        )

    def test_scoring(self):
        encoder = DeterministicCrossEncoder()

        self.assertEqual(encoder.score("machine", "machine guide"), 1.0)

    def test_score_batch(self):
        encoder = DeterministicCrossEncoder()

        self.assertEqual(
            encoder.score_batch(
                "machine",
                ["machine guide", "machine machine"],
            ),
            [1.0, 2.0],
        )

    def test_reranking_uses_document_text(self):
        candidates = [
            RetrievalResult("1", 0.9),
            RetrievalResult("2", 0.8),
        ]

        results = self.reranker.rerank("machine", candidates)

        self.assertEqual(
            [result.document_id for result in results],
            ["2", "1"],
        )
        self.assertEqual([result.score for result in results], [2.0, 1.0])

    def test_top_k(self):
        candidates = [
            RetrievalResult("1", 0.9),
            RetrievalResult("2", 0.8),
            RetrievalResult("3", 0.7),
        ]

        self.assertEqual(
            len(self.reranker.rerank("machine", candidates, top_k=1)),
            1,
        )

    def test_equal_scores_are_deterministic(self):
        candidates = [
            RetrievalResult("1", 0.9),
            RetrievalResult("4", 0.8),
        ]

        results = self.reranker.rerank("machine", candidates)

        self.assertEqual(
            [result.document_id for result in results],
            ["1", "4"],
        )

    def test_empty_candidates(self):
        self.assertEqual(
            self.reranker.rerank("machine", []),
            [],
        )

    def test_invalid_input(self):
        with self.assertRaises(TypeError):
            self.reranker.rerank(123, [])

        with self.assertRaises(TypeError):
            self.reranker.rerank("machine", ["not a result"])

        with self.assertRaises(ValueError):
            self.reranker.rerank("machine", [], top_k=0)

    def test_retrieval_candidates_can_be_reranked(self):
        analyzer = Analyzer()
        builder = IndexBuilder(
            analyzer,
            Vocabulary(),
            InvertedIndex(),
        )
        for document in self.corpus:
            builder.add_document(document)

        retriever = BM25Retriever(
            CandidateGenerator(builder.inverted_index),
            BM25(builder.forward_index, builder.statistics),
        )
        candidates = [
            RetrievalResult(str(document_id), score)
            for document_id, score in retriever.retrieve(
                analyzer.analyze("machine")
            )
        ]

        results = self.reranker.rerank("machine", candidates, top_k=2)

        self.assertEqual([result.document_id for result in results], ["2", "1"])


class TestSentenceTransformerCrossEncoder(unittest.TestCase):
    @patch("reranking.cross_encoder.SentenceTransformersModel")
    def test_adapter_normalizes_model_scores(self, model_class):
        model_class.return_value.predict.return_value = [
            0.25,
            0.75,
        ]
        encoder = SentenceTransformerCrossEncoder("local-test-model")

        self.assertEqual(
            encoder.score_batch("query", ["first", "second"]),
            [0.25, 0.75],
        )


if __name__ == "__main__":
    unittest.main()