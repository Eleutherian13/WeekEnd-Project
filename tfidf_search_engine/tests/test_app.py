import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import app
from document.corpus import Corpus
from search.search_engine import SearchEngine


class TestApplicationLayer(unittest.TestCase):
    def test_application_initialization_loads_corpus_and_builds_engine(self):
        corpus = app.load_corpus()
        engine = app.create_search_engine(corpus)

        self.assertIsInstance(corpus, Corpus)
        self.assertEqual(len(corpus), 6)
        self.assertIsInstance(engine, SearchEngine)

    def test_loader_rejects_malformed_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "documents.json"
            path.write_text(
                json.dumps([{"document_id": 1}]),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "record 1"):
                app.load_corpus(path)

    def test_valid_and_multi_term_queries_return_deterministic_results(self):
        engine = app.create_search_engine(app.load_corpus())

        single_term = engine.search("machine")
        multi_term = engine.search("machine learning")

        self.assertEqual(
            [document_id for document_id, _ in single_term],
            [1, 3],
        )
        self.assertEqual(
            [document_id for document_id, _ in multi_term],
            [1, 3, 2],
        )
        self.assertEqual(multi_term, engine.search("machine learning"))

    def test_no_result_and_invalid_queries(self):
        engine = app.create_search_engine(app.load_corpus())

        self.assertEqual(engine.search("quantum"), [])
        with self.assertRaises(ValueError):
            engine.search("")
        with self.assertRaises(ValueError):
            engine.search("   ")

    def test_top_k_limits_application_results(self):
        engine = app.create_search_engine(app.load_corpus())

        results = engine.search("machine learning", top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual([document_id for document_id, _ in results], [1, 3])

    def test_run_handles_repeated_queries_and_clean_exit(self):
        output = io.StringIO()
        input_stream = io.StringIO("1\nmachine\n2\n1\ncats\n2\n7\n")

        with patch("sys.stdin", input_stream), redirect_stdout(output):
            app.run()

        rendered = output.getvalue()
        self.assertIn("Documents loaded: 6", rendered)
        self.assertIn("Indexed documents: 6", rendered)
        self.assertIn("Machine learning is amazing", rendered)
        self.assertIn("Cats chase mice", rendered)
        self.assertIn("Goodbye.", rendered)

    def test_run_handles_blank_input_and_eof(self):
        blank_output = io.StringIO()
        with patch("sys.stdin", io.StringIO("1\n\n7\n")), redirect_stdout(blank_output):
            app.run()

        self.assertIn("Please enter a query.", blank_output.getvalue())
        self.assertIn("Goodbye.", blank_output.getvalue())

        eof_output = io.StringIO()
        with patch("sys.stdin", io.StringIO("")), redirect_stdout(eof_output):
            app.run()

        self.assertIn("Goodbye.", eof_output.getvalue())


if __name__ == "__main__":
    unittest.main()
