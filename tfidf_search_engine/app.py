from index.inverted_index import InvertedIndex
from index.posting import Posting
from index.posting_list import PostingList
from index.builder import IndexBuilder
from index.vocabulary import Vocabulary
from retieval.candidate_generator import CandidateGenerator
from retieval.lexical.retriever import LexicalRetriever
from analysis.analyzer import Analyzer


def test_candidate_generator():
    print("\n=== CANDIDATE GENERATOR TEST ===")

    # ---------------------------------------------------------
    # 1. Build a small inverted index from scratch
    # ---------------------------------------------------------

    inverted_index = InvertedIndex()

    # machine -> documents 1 and 3
    inverted_index.add(
        "machine",
        Posting(document_id=1, term_frequency=1),
    )

    inverted_index.add(
        "machine",
        Posting(document_id=3, term_frequency=1),
    )

    # learning -> documents 1, 2 and 3
    inverted_index.add(
        "learning",
        Posting(document_id=1, term_frequency=1),
    )

    inverted_index.add(
        "learning",
        Posting(document_id=2, term_frequency=1),
    )

    inverted_index.add(
        "learning",
        Posting(document_id=3, term_frequency=1),
    )

    # ---------------------------------------------------------
    # 2. Create CandidateGenerator
    # ---------------------------------------------------------

    candidate_generator = CandidateGenerator(
        inverted_index=inverted_index
    )

    # ---------------------------------------------------------
    # 3. Single-term candidate generation
    # ---------------------------------------------------------

    candidates = candidate_generator.generate(["machine"])

    print("machine candidates:")
    print(candidates)

    assert candidates == {1, 3}

    print("✓ Single-term candidate generation passed")

    # ---------------------------------------------------------
    # 4. Multi-term candidate generation
    # ---------------------------------------------------------

    candidates = candidate_generator.generate(
        ["machine", "learning"]
    )

    print("\nmachine + learning candidates:")
    print(candidates)

    assert candidates == {1, 2, 3}

    print("✓ Multi-term OR candidate generation passed")

    # ---------------------------------------------------------
    # 5. Unknown term
    # ---------------------------------------------------------

    candidates = candidate_generator.generate(
        ["nonexistent"]
    )

    print("\nnonexistent candidates:")
    print(candidates)

    assert candidates == set()

    print("✓ Unknown-term handling passed")

    # ---------------------------------------------------------
    # 6. Known + unknown term
    # ---------------------------------------------------------

    candidates = candidate_generator.generate(
        ["machine", "nonexistent"]
    )

    print("\nmachine + nonexistent candidates:")
    print(candidates)

    assert candidates == {1, 3}

    print("✓ Known + unknown term handling passed")

    # ---------------------------------------------------------
    # 7. Empty query
    # ---------------------------------------------------------

    candidates = candidate_generator.generate([])

    print("\nempty query candidates:")
    print(candidates)

    assert candidates == set()

    print("✓ Empty-query handling passed")

    # ---------------------------------------------------------
    # 8. Whitespace handling
    # ---------------------------------------------------------

    candidates = candidate_generator.generate(
        [" machine "]
    )

    print("\nwhitespace-normalized term candidates:")
    print(candidates)

    assert candidates == {1, 3}

    print("✓ Whitespace handling passed")

    # ---------------------------------------------------------
    # 9. Empty strings
    # ---------------------------------------------------------

    candidates = candidate_generator.generate(
        ["", "   ", "machine"]
    )

    print("\nempty-string candidates:")
    print(candidates)

    assert candidates == {1, 3}

    print("✓ Empty-string handling passed")

    # ---------------------------------------------------------
    # 10. Callable interface
    # ---------------------------------------------------------

    candidates = candidate_generator(
        ["learning"]
    )

    print("\ncallable interface candidates:")
    print(candidates)

    assert candidates == {1, 2, 3}

    print("✓ Callable interface passed")

    # ---------------------------------------------------------
    # 11. None validation
    # ---------------------------------------------------------

    try:
        candidate_generator.generate(None)
        assert False, "Expected TypeError for None"
    except TypeError:
        print("✓ None validation passed")

    # ---------------------------------------------------------
    # 12. Invalid query term validation
    # ---------------------------------------------------------

    try:
        candidate_generator.generate(
            ["machine", 123]
        )
        assert False, "Expected TypeError for non-string term"
    except TypeError:
        print("✓ Invalid-term validation passed")

    # ---------------------------------------------------------
    # 13. Constructor validation
    # ---------------------------------------------------------

    try:
        CandidateGenerator(None)
        assert False, "Expected TypeError for invalid inverted index"
    except TypeError:
        print("✓ Constructor validation passed")

    print("\n✓ ALL CANDIDATE GENERATOR TESTS PASSED")


class DummyLexicalRetriever(LexicalRetriever):
    """
    Temporary test implementation used only to verify
    the LexicalRetriever abstraction.
    """

    def retrieve(self, query_terms):
        candidates = self.candidate_generator.generate(query_terms)

        return [
            (document_id, 1.0)
            for document_id in sorted(candidates)
        ]


def test_lexical_retriever(builder):
    candidate_generator = CandidateGenerator(
        inverted_index=builder.inverted_index
    )

    retriever = DummyLexicalRetriever(
        candidate_generator=candidate_generator
    )

    results = retriever.retrieve(
        ["machine", "learning"]
    )

    print("Retriever results:", results)

    assert results == [
        (1, 1.0),
        (2, 1.0),
        (3, 1.0),
    ]

    # Test callable interface.
    results = retriever(["machine"])

    assert results == [
        (1, 1.0),
        (3, 1.0),
    ]

    print("LexicalRetriever contract tests passed.")


if __name__ == "__main__":
    test_candidate_generator()
    
    # Create a simple builder for testing lexical retriever
    analyzer = Analyzer()
    vocabulary = Vocabulary()
    inverted_index = InvertedIndex()
    builder = IndexBuilder(
        analyzer=analyzer,
        vocabulary=vocabulary,
        inverted_index=inverted_index
    )
    
    # Add sample documents to the builder's inverted index
    # Document 1: contains "machine" and "learning"
    inverted_index.add("machine", Posting(document_id=1, term_frequency=1))
    inverted_index.add("learning", Posting(document_id=1, term_frequency=1))
    
    # Document 2: contains "learning"
    inverted_index.add("learning", Posting(document_id=2, term_frequency=1))
    
    # Document 3: contains "machine" and "learning"
    inverted_index.add("machine", Posting(document_id=3, term_frequency=1))
    inverted_index.add("learning", Posting(document_id=3, term_frequency=1))
    
    test_lexical_retriever(builder)
