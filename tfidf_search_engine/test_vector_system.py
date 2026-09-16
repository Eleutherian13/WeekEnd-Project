"""
Comprehensive tests for VectorStore and HNSW.

Expected project structure:
    vector/
        __init__.py
        similarity.py
        embedding.py
        vector_store.py
        hnsw.py

Run:
    python -m unittest test_vector_system.py -v
"""

import random
import unittest

from vector.vector_store import VectorStore
from vector.hnsw import HNSW


class TestVectorStore(unittest.TestCase):

    def test_add_get_and_count(self):
        store = VectorStore(metric="cosine")
        store.add("a", [1.0, 0.0], {"type": "A"})

        self.assertEqual(store.count(), 1)
        self.assertTrue(store.contains("a"))

        vector, metadata = store.get("a")
        self.assertEqual(vector, [1.0, 0.0])
        self.assertEqual(metadata, {"type": "A"})

    def test_duplicate_id_rejected(self):
        store = VectorStore()
        store.add("a", [1.0, 0.0])

        with self.assertRaises(ValueError):
            store.add("a", [0.0, 1.0])

    def test_dimension_mismatch_rejected(self):
        store = VectorStore()
        store.add("a", [1.0, 0.0])

        with self.assertRaises(ValueError):
            store.add("b", [1.0, 0.0, 2.0])

    def test_update_vector_and_metadata(self):
        store = VectorStore()
        store.add("a", [1.0, 0.0], {"old": True})

        store.update("a", vector=[0.0, 1.0], metadata={"new": True})

        self.assertEqual(store.get_vector("a"), [0.0, 1.0])
        self.assertEqual(store.get_metadata("a"), {"new": True})

    def test_delete(self):
        store = VectorStore()
        store.add("a", [1.0, 0.0])
        store.delete("a")

        self.assertFalse(store.contains("a"))
        self.assertEqual(store.count(), 0)

        with self.assertRaises(KeyError):
            store.get_vector("a")

    def test_batch_is_atomic_for_duplicate_ids(self):
        store = VectorStore()
        store.add("existing", [1.0, 0.0])

        with self.assertRaises(ValueError):
            store.add_batch([
                ("a", [0.0, 1.0], None),
                ("existing", [0.5, 0.5], None),
            ])

        self.assertFalse(store.contains("a"))
        self.assertTrue(store.contains("existing"))

    def test_cosine_search(self):
        store = VectorStore(metric="cosine")
        store.add("right", [1.0, 0.0])
        store.add("up", [0.0, 1.0])
        store.add("diagonal", [0.8, 0.2])

        results = store.search([1.0, 0.0], k=3)

        self.assertTrue(all(hasattr(result, "id") for result in results))
        self.assertTrue(all(isinstance(result.score, float) for result in results))
        self.assertEqual(results[0].id, "right")
        self.assertGreaterEqual(results[0].score, results[1].score)

    def test_search_validates_k_and_query_vector(self):
        store = VectorStore()
        store.add("a", [1.0, 0.0])

        with self.assertRaises(ValueError):
            store.search([1.0, 0.0], k=0)

        with self.assertRaises(ValueError):
            store.search([1.0, 0.0, 0.0])

        with self.assertRaises(TypeError):
            store.search([1.0, 0.0], k=True)

    def test_dot_search(self):
        store = VectorStore(metric="dot")
        store.add("small", [1.0, 0.0])
        store.add("large", [10.0, 0.0])

        results = store.search([1.0, 0.0], k=2)

        self.assertEqual(results[0].id, "large")

    def test_euclidean_search(self):
        store = VectorStore(metric="euclidean")
        store.add("near", [1.0, 1.0])
        store.add("far", [10.0, 10.0])

        results = store.search([0.0, 0.0], k=2)

        self.assertEqual(results[0].id, "near")
        self.assertLessEqual(results[0].score, results[1].score)

    def test_metadata_filter(self):
        store = VectorStore()
        store.add("a", [1.0, 0.0], {"category": "x"})
        store.add("b", [0.9, 0.1], {"category": "y"})

        results = store.search(
            [1.0, 0.0],
            k=10,
            metadata_filter={"category": "y"},
        )

        self.assertEqual([result.id for result in results], ["b"])

    def test_clear_preserves_configured_dimension(self):
        store = VectorStore(dimension=2)
        store.add("a", [1.0, 0.0])
        store.clear()

        self.assertEqual(store.count(), 0)
        self.assertEqual(store.dimension, 2)


class TestHNSW(unittest.TestCase):

    def make_store_and_index(
        self,
        metric="cosine",
        m=4,
        ef_construction=20,
        ef_search=20,
        seed=42,
    ):
        store = VectorStore(metric=metric)
        index = HNSW(
            store,
            m=m,
            ef_construction=ef_construction,
            ef_search=ef_search,
            metric=metric,
            seed=seed,
        )
        return store, index

    def add_both(self, store, index, vector_id, vector):
        store.add(vector_id, vector)
        index.add(vector_id, vector)

    def assert_graph_invariants(self, index):
        # Every edge must be bidirectional and respect layer existence.
        for node_id, node in index._nodes.items():
            self.assertEqual(len(node.neighbors), node.level + 1)

            for level, neighbors in enumerate(node.neighbors):
                self.assertLessEqual(
                    len(neighbors),
                    index._max_neighbors(level),
                    msg=f"{node_id} exceeds degree limit at level {level}",
                )

                for neighbor_id in neighbors:
                    self.assertIn(
                        neighbor_id,
                        index._nodes,
                        msg=f"{node_id} points to missing node {neighbor_id}",
                    )

                    neighbor = index._nodes[neighbor_id]
                    self.assertGreaterEqual(
                        neighbor.level,
                        level,
                        msg=f"{neighbor_id} does not exist at layer {level}",
                    )

                    self.assertIn(
                        node_id,
                        neighbor.neighbors[level],
                        msg=(
                            f"Edge {node_id} -> {neighbor_id} "
                            f"is not bidirectional at layer {level}"
                        ),
                    )

        if index._nodes:
            self.assertIsNotNone(index._entry_point)
            self.assertEqual(
                index._max_level,
                max(node.level for node in index._nodes.values()),
            )
            self.assertEqual(
                index._nodes[index._entry_point].level,
                index._max_level,
            )

    def test_empty_search(self):
        store, index = self.make_store_and_index()
        self.assertEqual(index.search([1.0, 0.0]), [])
        self.assertEqual(len(index), 0)

    def test_single_node(self):
        store, index = self.make_store_and_index()
        self.add_both(store, index, "a", [1.0, 0.0])

        results = index.search([1.0, 0.0], k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "a")
        self.assert_graph_invariants(index)

    def test_basic_insertion_and_graph_integrity(self):
        store, index = self.make_store_and_index()

        vectors = {
            "A": [1.0, 0.0],
            "B": [0.9, 0.1],
            "C": [0.7, 0.7],
            "D": [0.0, 1.0],
            "E": [-0.7, 0.7],
            "F": [-1.0, 0.0],
        }

        for vector_id, vector in vectors.items():
            self.add_both(store, index, vector_id, vector)
            self.assertEqual(len(index), store.count())
            self.assert_graph_invariants(index)

    def test_duplicate_hnsw_id_rejected(self):
        store, index = self.make_store_and_index()
        self.add_both(store, index, "a", [1.0, 0.0])

        with self.assertRaises(ValueError):
            index.add("a", [1.0, 0.0])

    def test_dimension_mismatch_rejected(self):
        store, index = self.make_store_and_index()
        self.add_both(store, index, "a", [1.0, 0.0])

        with self.assertRaises(ValueError):
            index.add("bad", [1.0, 0.0, 2.0])

    def test_known_query(self):
        store, index = self.make_store_and_index()

        vectors = {
            "right": [1.0, 0.0],
            "almost_right": [0.95, 0.05],
            "up": [0.0, 1.0],
            "left": [-1.0, 0.0],
        }

        for vector_id, vector in vectors.items():
            self.add_both(store, index, vector_id, vector)

        results = index.search([1.0, 0.0], k=1, ef_search=20)

        self.assertEqual(results[0].id, "right")

    def test_hnsw_vs_exact_top1_on_small_dataset(self):
        store, index = self.make_store_and_index(
            m=4,
            ef_construction=50,
            ef_search=50,
        )

        vectors = {
            "A": [1.0, 0.0],
            "B": [0.9, 0.1],
            "C": [0.7, 0.7],
            "D": [0.0, 1.0],
            "E": [-0.7, 0.7],
            "F": [-1.0, 0.0],
        }

        for vector_id, vector in vectors.items():
            self.add_both(store, index, vector_id, vector)

        queries = [
            [1.0, 0.0],
            [0.8, 0.6],
            [0.0, 1.0],
            [-0.8, 0.2],
        ]

        for query in queries:
            exact = store.search(query, k=1)[0].id
            approximate = index.search(query, k=1, ef_search=50)[0].id
            self.assertEqual(
                approximate,
                exact,
                msg=f"HNSW missed exact nearest neighbor for {query}",
            )

    def test_remove_non_entry_node(self):
        store, index = self.make_store_and_index()

        for vector_id, vector in {
            "a": [1.0, 0.0],
            "b": [0.0, 1.0],
            "c": [-1.0, 0.0],
        }.items():
            self.add_both(store, index, vector_id, vector)

        index.remove("b")

        self.assertFalse(index.contains("b"))
        self.assertEqual(len(index), 2)

        for node in index._nodes.values():
            for neighbors in node.neighbors:
                self.assertNotIn("b", neighbors)

        self.assert_graph_invariants(index)

    def test_remove_entry_point(self):
        store, index = self.make_store_and_index(seed=7)

        for i in range(12):
            vector_id = f"v{i}"
            vector = [float(i), float(i * i + 1)]
            self.add_both(store, index, vector_id, vector)

        old_entry = index._entry_point
        index.remove(old_entry)

        self.assertNotIn(old_entry, index._nodes)
        self.assertIsNotNone(index._entry_point)
        self.assert_graph_invariants(index)

    def test_remove_all_nodes_resets_index(self):
        store, index = self.make_store_and_index()
        self.add_both(store, index, "a", [1.0, 0.0])

        index.remove("a")

        self.assertEqual(len(index), 0)
        self.assertIsNone(index._entry_point)
        self.assertEqual(index._max_level, -1)
        self.assertIsNone(index._dimension)

    def test_clear(self):
        store, index = self.make_store_and_index()

        for i in range(5):
            self.add_both(store, index, str(i), [float(i), float(i + 1)])

        index.clear()

        self.assertEqual(len(index), 0)
        self.assertIsNone(index._entry_point)
        self.assertEqual(index._max_level, -1)
        self.assertIsNone(index._dimension)

    def test_euclidean_metric(self):
        store, index = self.make_store_and_index(
            metric="euclidean",
            ef_construction=50,
            ef_search=50,
        )

        vectors = {
            "near": [1.0, 1.0],
            "middle": [3.0, 3.0],
            "far": [10.0, 10.0],
        }

        for vector_id, vector in vectors.items():
            self.add_both(store, index, vector_id, vector)

        exact = store.search([0.0, 0.0], k=1)[0].id
        approximate = index.search([0.0, 0.0], k=1)[0].id

        self.assertEqual(exact, "near")
        self.assertEqual(approximate, exact)
        self.assert_graph_invariants(index)

    def test_random_dataset_recall(self):
        rng = random.Random(123)

        store, index = self.make_store_and_index(
            m=8,
            ef_construction=80,
            ef_search=80,
            seed=123,
        )

        dimension = 8
        n_vectors = 200
        queries = 30
        k = 5

        for i in range(n_vectors):
            vector = [
                rng.uniform(-1.0, 1.0)
                for _ in range(dimension)
            ]
            vector_id = f"v{i}"
            self.add_both(store, index, vector_id, vector)

        self.assert_graph_invariants(index)

        recalls = []

        for _ in range(queries):
            query = [
                rng.uniform(-1.0, 1.0)
                for _ in range(dimension)
            ]

            exact_ids = {
                result.id
                for result in store.search(query, k=k)
            }

            approximate_ids = {
                result.id
                for result in index.search(
                    query,
                    k=k,
                    ef_search=80,
                )
            }

            recall = len(exact_ids & approximate_ids) / k
            recalls.append(recall)

        average_recall = sum(recalls) / len(recalls)

        # This is intentionally a practical sanity threshold, not a
        # mathematical guarantee for every random graph realization.
        self.assertGreaterEqual(
            average_recall,
            0.70,
            msg=f"Average recall@{k} too low: {average_recall:.3f}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)