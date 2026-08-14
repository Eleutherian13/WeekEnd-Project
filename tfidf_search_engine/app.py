from index.inverted_index import InvertedIndex
from index.posting import Posting
from index.posting_list import PostingList
from retieval.candidate_generator import CandidateGenerator


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

if __name__ == "__main__":
    test_candidate_generator()