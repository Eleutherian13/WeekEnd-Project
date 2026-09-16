import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import app
from planning import RetrievalMode
from rag import RAGContext
from retieval.lexical.retriever import RetrievalResult
from search.search_engine import GenerationFailure, RAGResponse


class FakePlanner:
    def plan(self, query, top_k):
        return type("Plan", (), {"enabled_modes": ()})()


class FakeDense:
    def retrieve(self, query, k):
        return [RetrievalResult("1", 0.9)]


class FakeReranker:
    def rerank(self, query, candidates, top_k):
        return [RetrievalResult("1", 1.5)]


class FakeEngine:
    dense_retriever = FakeDense()
    graph_retriever = None
    reranker = FakeReranker()
    generator = object()
    planner = FakePlanner()

    def retrieve(self, query, top_k):
        return [RetrievalResult("1", 0.9)]

    def answer(self, query, top_k):
        return RAGResponse(
            query=query,
            results=(RetrievalResult("1", 1.5),),
            evidence=(),
            context=RAGContext(query, "context", ()),
            answer="grounded answer",
        )

    def search(self, query, top_k=10):
        return [(1, 0.9)]


class TestCliModes(unittest.TestCase):
    def fake_runtime(self):
        return app.Runtime(
            corpus=app.load_corpus(),
            engine=FakeEngine(),
        )

    def test_menu_reports_real_capabilities(self):
        runtime = app.create_runtime(app.load_corpus())
        output = io.StringIO()

        with redirect_stdout(output):
            app.print_menu(runtime)

        rendered = output.getvalue()
        self.assertIn("1. Lexical BM25 - AVAILABLE", rendered)
        self.assertIn("3. Graph Retrieval - NOT CONFIGURED", rendered)
        self.assertIn("4. Hybrid Retrieval - AVAILABLE", rendered)

    def test_lexical_mode_displays_ranked_documents(self):
        runtime = app.Runtime(
            corpus=app.load_corpus(),
            engine=app.create_search_engine(app.load_corpus()),
        )
        output = io.StringIO()

        with patch("builtins.input", side_effect=["machine", "2"]), redirect_stdout(output):
            app._run_mode(runtime, "1")

        rendered = output.getvalue()
        self.assertIn("Document ID:", rendered)
        self.assertIn("Machine learning is amazing", rendered)
        self.assertIn("Score:", rendered)

    def test_graph_mode_reports_unconfigured(self):
        runtime = app.Runtime(
            corpus=app.load_corpus(),
            engine=app.create_search_engine(app.load_corpus()),
        )
        output = io.StringIO()

        with redirect_stdout(output):
            app._run_mode(runtime, "3")

        self.assertIn("NOT CONFIGURED", output.getvalue())
        self.assertIn("no graph data source", output.getvalue())

    def test_dense_mode_displays_results(self):
        output = io.StringIO()
        with patch("builtins.input", side_effect=["machine", "1"]), redirect_stdout(output):
            app._run_mode(self.fake_runtime(), "2")

        self.assertIn("Document ID: 1", output.getvalue())

    def test_hybrid_mode_displays_used_modes_and_results(self):
        output = io.StringIO()
        runtime = self.fake_runtime()
        runtime.engine.planner = type(
            "Planner",
            (),
            {"plan": lambda self, query, top_k: type(
                "Plan", (), {"enabled_modes": (
                    RetrievalMode.LEXICAL,
                    RetrievalMode.DENSE,
                )}
            )()},
        )()
        with patch("builtins.input", side_effect=["machine", "1"]), redirect_stdout(output):
            app._run_mode(runtime, "4")

        self.assertIn("Modes used: lexical, dense", output.getvalue())

    def test_reranking_mode_displays_both_scores(self):
        output = io.StringIO()
        with patch("builtins.input", side_effect=["machine", "1"]), redirect_stdout(output):
            app._run_mode(self.fake_runtime(), "5")

        self.assertIn("Retrieval score:", output.getvalue())
        self.assertIn("Reranking score:", output.getvalue())

    def test_rag_mode_displays_answer(self):
        output = io.StringIO()
        with patch("builtins.input", side_effect=["machine", "1"]), redirect_stdout(output):
            app._run_mode(self.fake_runtime(), "6")

        self.assertIn("Answer:", output.getvalue())
        self.assertIn("grounded answer", output.getvalue())

    def test_invalid_mode_and_top_k_are_reported(self):
        runtime = app.Runtime(
            corpus=app.load_corpus(),
            engine=app.create_search_engine(app.load_corpus()),
        )
        output = io.StringIO()

        with patch("sys.stdin", io.StringIO("9\n1\nmachine\nnope\n7\n")), redirect_stdout(output):
            app.run()

        rendered = output.getvalue()
        self.assertIn("Invalid mode", rendered)
        self.assertIn("Invalid top_k", rendered)

    def test_rag_generation_failure_is_reported(self):
        runtime = self.fake_runtime()
        output = io.StringIO()

        with patch.object(
            runtime.engine,
            "answer",
            side_effect=GenerationFailure("generator unavailable"),
        ), patch("builtins.input", side_effect=["machine", "1"]), redirect_stdout(output):
            app._run_mode(runtime, "6")

        self.assertIn("Generation failed: generator unavailable", output.getvalue())


if __name__ == "__main__":
    unittest.main()