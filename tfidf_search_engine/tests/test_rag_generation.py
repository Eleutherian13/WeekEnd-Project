import os
import unittest
from unittest.mock import patch

from analysis.analyzer import Analyzer
from document.corpus import Corpus
from document.documents import Document
from index.builder import IndexBuilder
from index.inverted_index import InvertedIndex
from index.vocabulary import Vocabulary
from ranking.bm25 import BM25
from rag import ContextBuilder, EvidenceBuilder, OllamaGenerator
from retieval.candidate_generator import CandidateGenerator
from retieval.lexical.bm25_retriever import BM25Retriever
from retieval.lexical.retriever import RetrievalResult


class FakeOllamaClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class TestRAGGeneration(unittest.TestCase):
    def setUp(self) -> None:
        self.corpus = Corpus()
        self.corpus.add(Document(1, "Machine learning uses neural networks."))
        self.corpus.add(Document(2, "Cats chase mice."))

        analyzer = Analyzer()
        builder = IndexBuilder(
            analyzer,
            Vocabulary(),
            InvertedIndex(),
        )
        for document in self.corpus:
            builder.add_document(document)

        self.retriever = BM25Retriever(
            CandidateGenerator(builder.inverted_index),
            BM25(builder.forward_index, builder.statistics),
        )

    def _build_context(self):
        candidates = [
            RetrievalResult(str(document_id), score)
            for document_id, score in self.retriever.retrieve(
                Analyzer().analyze("machine learning")
            )
        ]
        evidence = EvidenceBuilder(
            lambda document_id: self.corpus[int(document_id)].text
        ).build(candidates)
        return ContextBuilder().build("machine learning", list(evidence))

    def test_generator_receives_rag_context_from_retrieval(self):
        client = FakeOllamaClient({"response": "Neural networks are used."})
        generator = OllamaGenerator(
            model_name="test-model",
            client=client,
        )

        answer = generator.generate(self._build_context())

        self.assertEqual(answer, "Neural networks are used.")
        self.assertEqual(len(client.calls), 1)
        self.assertIn("Machine learning uses neural networks.", client.calls[0]["prompt"])
        self.assertNotIn("Cats chase mice.", client.calls[0]["prompt"])
        self.assertEqual(client.calls[0]["model"], "test-model")

    def test_empty_context_is_rejected(self):
        context = ContextBuilder().build("machine", [])
        generator = OllamaGenerator(
            model_name="test-model",
            client=FakeOllamaClient({"response": "should not run"}),
        )

        with self.assertRaises(ValueError):
            generator.generate(context)

    def test_missing_model_is_rejected(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                OllamaGenerator(client=FakeOllamaClient())

    def test_missing_provider_is_rejected(self):
        with patch("rag.generator.ollama", None):
            with self.assertRaises(RuntimeError):
                OllamaGenerator(model_name="test-model")

    def test_timeout_is_reported(self):
        generator = OllamaGenerator(
            model_name="test-model",
            client=FakeOllamaClient(error=TimeoutError()),
        )

        with self.assertRaises(TimeoutError):
            generator.generate(self._build_context())

    def test_provider_error_is_reported(self):
        generator = OllamaGenerator(
            model_name="test-model",
            client=FakeOllamaClient(error=ConnectionError("offline")),
        )

        with self.assertRaises(RuntimeError):
            generator.generate(self._build_context())

    def test_invalid_provider_response_is_rejected(self):
        generator = OllamaGenerator(
            model_name="test-model",
            client=FakeOllamaClient({"unexpected": "payload"}),
        )

        with self.assertRaises(RuntimeError):
            generator.generate(self._build_context())

    def test_invalid_context_is_rejected(self):
        generator = OllamaGenerator(
            model_name="test-model",
            client=FakeOllamaClient(),
        )

        with self.assertRaises(TypeError):
            generator.generate("not context")


if __name__ == "__main__":
    unittest.main()