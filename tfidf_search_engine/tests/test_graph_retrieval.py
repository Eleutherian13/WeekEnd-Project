import unittest

from graph.edge import Edge
from graph.graph import Graph
from graph.node import Node
from retieval.graph.graph_retriever import SimpleGraphRetriever
from retieval.lexical.retriever import RetrievalResult


class TestGraphRetrieval(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = Graph()
        self.graph.add_node(
            Node("python", "entity", {"name": "Python"})
        )
        self.graph.add_node(
            Node("programming", "topic", {"name": "programming"})
        )
        self.graph.add_node(
            Node("document-one", "document", {"title": "Python guide"})
        )
        self.graph.add_node(
            Node("document-two", "document", {"title": "Python history"})
        )
        self.graph.add_edge(Edge("python", "document-one", "describes"))
        self.graph.add_edge(Edge("python", "programming", "related-to"))
        self.graph.add_edge(
            Edge("programming", "document-two", "supports")
        )
        self.retriever = SimpleGraphRetriever(
            self.graph,
            max_hops=2,
        )

    def test_query_matches_entity_node(self):
        self.assertEqual(
            self.retriever.match_nodes("python"),
            ("python",),
        )

    def test_one_hop_retrieval_returns_document(self):
        results = self.retriever.retrieve("python", top_k=1)

        self.assertEqual([result.document_id for result in results], [
            "document-one",
        ])
        self.assertEqual(results[0].score, 0.5)

    def test_multi_hop_retrieval_returns_second_document(self):
        results = self.retriever.retrieve("python", top_k=2)

        self.assertEqual(
            [result.document_id for result in results],
            ["document-one", "document-two"],
        )
        self.assertEqual(results[1].score, 1 / 3)

    def test_no_matching_node_returns_empty_results(self):
        self.assertEqual(
            self.retriever.retrieve("quantum", top_k=2),
            [],
        )

    def test_invalid_query_and_top_k_are_rejected(self):
        with self.assertRaises(TypeError):
            self.retriever.retrieve(123)

        with self.assertRaises(ValueError):
            self.retriever.retrieve("   ")

        with self.assertRaises(ValueError):
            self.retriever.retrieve("python", top_k=0)

    def test_top_k_limits_results(self):
        self.assertEqual(
            len(self.retriever.retrieve("python", top_k=1)),
            1,
        )

    def test_results_use_retrieval_result_contract(self):
        results = self.retriever.retrieve("python")

        self.assertTrue(
            all(isinstance(result, RetrievalResult) for result in results)
        )
        self.assertEqual(
            results[0].metadata,
            {"title": "Python guide"},
        )


if __name__ == "__main__":
    unittest.main()