import unittest

from document.documents import Document
from retieval.dense.retriever import DenseRetriever
from retieval.lexical.retriever import RetrievalResult
from vector.embedding import EmbeddingModel
from vector.similarity import VectorSimilarity
from vector.vector_store import VectorStore


class DeterministicEmbedding(EmbeddingModel):
    _vectors = {
        "machine learning": [1.0, 0.0],
        "machine vision": [0.0, 1.0],
        "query": [1.0, 0.0],
    }

    def embed(self, text: str) -> list[float]:
        text = self._validate_text(text)
        return self._validate_embedding(
            self._vectors.get(text, [0.0, 1.0])
        )


class TestDenseRetrieval(unittest.TestCase):
    def setUp(self) -> None:
        self.embedding = DeterministicEmbedding()
        self.documents = [
            Document(1, "machine learning", {"topic": "learning"}),
            Document(2, "machine vision", {"topic": "vision"}),
        ]
        self.retriever = DenseRetriever.from_documents(
            self.documents,
            self.embedding,
        )

    def test_embedding_returns_valid_deterministic_vector(self):
        self.assertEqual(
            self.embedding.embed("query"),
            [1.0, 0.0],
        )
        self.assertEqual(
            self.embedding.embed_batch(["query", "machine vision"]),
            [[1.0, 0.0], [0.0, 1.0]],
        )

    def test_vector_insertion_uses_existing_store(self):
        store = self.retriever.vector_store

        self.assertEqual(store.count(), 2)
        self.assertEqual(store.get_vector("1"), [1.0, 0.0])
        self.assertEqual(
            store.get_metadata("1"),
            {"topic": "learning"},
        )

    def test_exact_similarity_search_returns_best_match(self):
        store = self.retriever.vector_store

        results = store.search([1.0, 0.0], k=2)

        self.assertEqual([result.id for result in results], ["1", "2"])
        self.assertEqual(results[0].score, 1.0)
        self.assertEqual(
            VectorSimilarity.cosine_similarity([1.0, 0.0], [1.0, 0.0]),
            1.0,
        )

    def test_dense_retrieval_returns_retrieval_results(self):
        results = self.retriever.retrieve("query")

        self.assertIsInstance(results[0], RetrievalResult)
        self.assertEqual(results[0].document_id, "1")
        self.assertEqual(results[0].metadata, {"topic": "learning"})

    def test_top_k_limits_dense_results(self):
        self.assertEqual(
            len(self.retriever.retrieve("query", k=1)),
            1,
        )

    def test_empty_store_returns_empty_results(self):
        retriever = DenseRetriever(VectorStore(self.embedding))

        self.assertEqual(retriever.retrieve("query"), [])

    def test_equal_scores_are_deterministic(self):
        store = VectorStore(self.embedding)
        store.add_text("first", "machine learning")
        store.add_text("second", "machine learning")
        retriever = DenseRetriever(store)

        first = retriever.retrieve("query", k=2)
        second = retriever.retrieve("query", k=2)

        self.assertEqual(first, second)
        self.assertEqual(
            [result.document_id for result in first],
            ["first", "second"],
        )


if __name__ == "__main__":
    unittest.main()