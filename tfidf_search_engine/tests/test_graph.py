import unittest

from graph.edge import Edge
from graph.graph import Graph
from graph.node import Node
from graph.store import GraphStore
from graph.traversal import GraphTraversal
from retieval.graph.graph_retriever import SimpleGraphRetriever
from retieval.lexical.retriever import RetrievalResult


class TestGraph(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = Graph()
        self.graph.add_node(Node("query", "query"))
        self.graph.add_node(Node("entity", "entity"))
        self.graph.add_node(Node("document", "document"))
        self.graph.add_edge(Edge("query", "entity", "mentions"))
        self.graph.add_edge(Edge("entity", "document", "supports"))

    def test_bfs_and_k_hop_neighbors(self):
        traversal = GraphTraversal(self.graph)

        self.assertEqual(
            tuple(traversal.bfs("query")),
            ("query", "entity", "document"),
        )
        self.assertEqual(
            traversal.k_hop_neighbors("query", 1),
            {"entity"},
        )

    def test_graph_counts_and_lookup(self):
        self.assertEqual(self.graph.node_count, 3)
        self.assertEqual(self.graph.edge_count, 2)
        self.assertTrue(self.graph.has_edge("query", "entity"))
        self.assertEqual(
            self.graph.get_node("document").node_type,
            "document",
        )

    def test_graph_store_delegates_to_graph(self):
        store = GraphStore(self.graph)

        self.assertEqual(store.node_count(), 3)
        self.assertEqual(store.edge_count(), 2)
        self.assertEqual(store.get_node("entity").node_type, "entity")

    def test_duplicate_nodes_are_rejected(self):
        with self.assertRaises(ValueError):
            self.graph.add_node(Node("query", "query"))

    def test_graph_retriever_returns_ranked_results(self):
        retriever = SimpleGraphRetriever(
            self.graph,
            start_nodes=["query"],
            max_hops=2,
        )

        results = retriever.retrieve("machine", top_k=1)

        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], RetrievalResult)
        self.assertEqual(results[0].document_id, "document")
        self.assertEqual(results[0].score, 1 / 3)


if __name__ == "__main__":
    unittest.main()
