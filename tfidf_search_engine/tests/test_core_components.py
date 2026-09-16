import unittest

from analysis.analyzer import Analyzer
from document.corpus import Corpus
from document.documents import Document
from index.builder import IndexBuilder
from index.inverted_index import InvertedIndex
from index.vocabulary import Vocabulary
from planning import QueryAnalyzer, QueryPlanner, RetrievalMode
from query.boolean_query import AndQuery, NotQuery, OrQuery, TermQuery
from query.phrase_query import PhraseQuery
from query.query import Query
from query.query_processing import QueryProcessor
from rag import ContextBuilder, Evidence, EvidenceBuilder
from ranking.bm25 import BM25
from ranking.fusion import ScoreFusion
from ranking.reranker import Reranker
from ranking.tfidf import TFIDF
from retieval.lexical.retriever import RetrievalResult
from reranking import CrossEncoder


class TestAnalysisAndDocuments(unittest.TestCase):
    def test_analyzer_returns_normalized_processed_terms(self):
        terms = Analyzer().analyze("The Machine!")

        self.assertIsInstance(terms, list)
        self.assertEqual(terms, ["machin"])

    def test_analyzer_rejects_non_text(self):
        with self.assertRaises(TypeError):
            Analyzer().analyze(123)

    def test_document_and_corpus_preserve_documents(self):
        document = Document(1, "machine learning", {"kind": "text"})
        corpus = Corpus()
        corpus.add(document)

        self.assertEqual(len(corpus), 1)
        self.assertIn(1, corpus)
        self.assertIs(corpus[1], document)
        self.assertEqual(document.metadata["kind"], "text")

    def test_document_and_corpus_reject_invalid_state(self):
        with self.assertRaises(TypeError):
            Document(1, 123)

        with self.assertRaises(ValueError):
            Document(-1, "text")

        corpus = Corpus()
        corpus.add(Document(1, "text"))
        with self.assertRaises(ValueError):
            corpus.add(Document(1, "duplicate"))


class TestIndexAndQuery(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = Analyzer()
        self.builder = IndexBuilder(
            self.analyzer,
            Vocabulary(),
            InvertedIndex(),
        )
        self.builder.add_document(
            Document(1, "machine learning systems")
        )
        self.builder.add_document(
            Document(2, "machine vision")
        )

    def test_builder_populates_all_lexical_indexes(self):
        machine = self.analyzer.analyze("machine")[0]
        learning = self.analyzer.analyze("learning")[0]

        self.assertEqual(
            self.builder.forward_index.get_term_frequency(1, machine),
            1,
        )
        self.assertEqual(
            self.builder.inverted_index.get_postings(machine).document_frequency,
            2,
        )
        self.assertEqual(
            self.builder.positional_index.get_positions(machine, 1),
            [0],
        )
        self.assertTrue(self.builder.vocabulary.contains(learning))
        self.assertEqual(self.builder.statistics.document_count(), 2)

    def test_query_processing_returns_analyzed_terms(self):
        processor = QueryProcessor(self.analyzer)

        self.assertEqual(
            processor.process(Query("Machine")),
            ["machin"],
        )

        with self.assertRaises(ValueError):
            Query("   ")

    def test_boolean_queries_return_document_sets(self):
        machine = self.analyzer.analyze("machine")[0]
        learning = self.analyzer.analyze("learning")[0]
        index = self.builder.inverted_index

        self.assertEqual(TermQuery(machine).evaluate(index), {1, 2})
        self.assertEqual(
            AndQuery(
                TermQuery(machine),
                TermQuery(learning),
            ).evaluate(index),
            {1},
        )
        self.assertEqual(
            OrQuery(
                TermQuery(machine),
                TermQuery(learning),
            ).evaluate(index),
            {1, 2},
        )
        self.assertEqual(
            NotQuery(TermQuery(machine), {1, 2, 3}).evaluate(index),
            {3},
        )

    def test_phrase_query_requires_consecutive_terms(self):
        machine = self.analyzer.analyze("machine")[0]
        learning = self.analyzer.analyze("learning")[0]

        self.assertEqual(
            PhraseQuery([machine, learning]).evaluate(
                self.builder.positional_index
            ),
            {1},
        )
        self.assertEqual(
            PhraseQuery([learning, machine]).evaluate(
                self.builder.positional_index
            ),
            set(),
        )


class TestRanking(unittest.TestCase):
    def setUp(self) -> None:
        analyzer = Analyzer()
        builder = IndexBuilder(analyzer, Vocabulary(), InvertedIndex())
        builder.add_document(1, "machine learning")
        builder.add_document(2, "machine")
        self.builder = builder
        self.machine = analyzer.analyze("machine")[0]

    def test_bm25_and_tfidf_return_numeric_scores(self):
        bm25 = BM25(
            self.builder.forward_index,
            self.builder.statistics,
        )
        tfidf = TFIDF(
            self.builder.forward_index,
            self.builder.statistics,
        )

        self.assertIsInstance(bm25.score(self.machine, 1), float)
        self.assertGreater(bm25.score(self.machine, 1), 0.0)
        self.assertIsInstance(tfidf.tfidf(self.machine, 1), float)
        self.assertGreater(tfidf.tfidf(self.machine, 1), 0.0)

    def test_score_fusion_normalizes_and_weights_signals(self):
        fusion = ScoreFusion({"lexical": 1.0})

        scores = fusion.fuse({
            "lexical": {1: 0.2, 2: 0.8},
        })

        self.assertEqual(scores, {1: 0.0, 2: 1.0})
        self.assertEqual(fusion.fuse({}), {})

    def test_reranker_orders_scores_and_applies_top_k(self):
        reranker = Reranker()

        self.assertEqual(
            reranker.rerank({2: 0.5, 1: 0.5, 3: 0.9}, top_k=2),
            [(3, 0.9), (1, 0.5)],
        )

        with self.assertRaises(TypeError):
            reranker.rerank({"1": 1.0})


class TestPlanningAndRAG(unittest.TestCase):
    def test_query_analyzer_and_planner_are_deterministic(self):
        analyzer = QueryAnalyzer()
        analysis = analyzer.analyze('"machine learning"')
        plan = QueryPlanner().plan("what explains machine learning", top_k=3)

        self.assertTrue(analysis.has_exact_phrase)
        self.assertEqual(analysis.terms, ("machine", "learning"))
        self.assertEqual(plan.final_k, 3)
        self.assertGreaterEqual(plan.candidate_k, plan.final_k)
        self.assertIn(RetrievalMode.DENSE, plan.enabled_modes)

    def test_planner_rejects_invalid_top_k(self):
        with self.assertRaises(ValueError):
            QueryPlanner().plan("machine", top_k=0)

    def test_evidence_and_context_preserve_provenance(self):
        results = [
            RetrievalResult("1", 0.9, {"source": "test"}),
            RetrievalResult("2", 0.8, None),
        ]
        evidence = EvidenceBuilder(
            lambda document_id: f"text for {document_id}"
        ).build(results)
        context = ContextBuilder().build("machine", evidence)

        self.assertEqual([item.document_id for item in evidence], ["1", "2"])
        self.assertIsInstance(context.text, str)
        self.assertIn("document_id: 1", context.text)
        self.assertEqual(len(context.evidence), 2)

    def test_context_builder_enforces_input_contract(self):
        with self.assertRaises(ValueError):
            ContextBuilder().build("", [])

        with self.assertRaises(TypeError):
            ContextBuilder().build("query", ["not evidence"])


class TestRerankingContract(unittest.TestCase):
    def test_cross_encoder_is_abstract_until_a_provider_exists(self):
        self.assertIn("score", CrossEncoder.__abstractmethods__)
        with self.assertRaises(TypeError):
            CrossEncoder()


if __name__ == "__main__":
    unittest.main()
