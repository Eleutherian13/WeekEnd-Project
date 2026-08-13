from analysis.analyzer import Analyzer
from index.builder import IndexBuilder
from index.vocabulary import Vocabulary
from index.inverted_index import InvertedIndex
from ranking.bm25 import BM25


analyzer = Analyzer()

vocabulary = Vocabulary()
inverted_index = InvertedIndex()

builder = IndexBuilder(
    analyzer=analyzer,
    vocabulary=vocabulary,
    inverted_index=inverted_index,
)


documents = {
    1: "machine learning is amazing",
    2: "deep learning powers artificial intelligence",
    3: "machine learning uses neural networks",
    4: "cats chase mice",
    5: "dogs chase cats",
    6: "deep neural networks power AI",
}


for document_id, text in documents.items():

    builder.add_document(
        document_id,
        text,
    )


bm25 = BM25(
    forward_index=builder.forward_index,
    statistics=builder.statistics,
)


query_terms = [
    "machine",
    "learning",
]


print("\nBM25 PARAMETERS")
print("----------------")
print("k1:", bm25.k1)
print("b:", bm25.b)


print("\nBM25 IDF")
print("--------")
print(
    "machine:",
    bm25.inverse_document_frequency(
        "machine"
    ),
)
print(
    "learning:",
    bm25.inverse_document_frequency(
        "learning"
    ),
)


print("\nTERM SCORES")
print("-----------")

for document_id in documents:

    machine_score = bm25.score(
        "machine",
        document_id,
    )

    learning_score = bm25.score(
        "learning",
        document_id,
    )

    print(
        f"Document {document_id}: "
        f"machine={machine_score:.4f}, "
        f"learning={learning_score:.4f}"
    )


print("\nDOCUMENT SCORES")
print("---------------")

scores = {}

for document_id in documents:

    score = bm25.score_document(
        query_terms,
        document_id,
    )

    scores[document_id] = score

    print(
        f"Document {document_id}: "
        f"{score:.4f}"
    )


print("\nRANKED DOCUMENTS")
print("----------------")

ranked_documents = sorted(
    scores.items(),
    key=lambda item: item[1],
    reverse=True,
)

for document_id, score in ranked_documents:

    print(
        f"Document {document_id}: "
        f"{score:.4f} | "
        f"{documents[document_id]}"
    )


import math

from ranking.fusion import ScoreFusion


def test_score_fusion() -> None:

    fusion = ScoreFusion(
        weights={
            "bm25": 0.7,
            "dense": 0.3,
        }
    )

    scores = {
        "bm25": {
            1: 2.0,
            2: 5.0,
            3: 8.0,
        },
        "dense": {
            1: 0.9,
            2: 0.6,
            3: 0.3,
        },
    }

    normalized_bm25 = fusion.normalize(
        scores["bm25"]
    )

    normalized_dense = fusion.normalize(
        scores["dense"]
    )

    fused_scores = fusion.fuse(
        scores
    )

    print("\n=== SCORE FUSION TEST ===")

    print("\nWeights:")
    print(fusion.weights)

    print("\nBM25 scores:")
    print(scores["bm25"])

    print("\nNormalized BM25:")
    print(normalized_bm25)

    print("\nDense scores:")
    print(scores["dense"])

    print("\nNormalized Dense:")
    print(normalized_dense)

    print("\nFused scores:")
    print(fused_scores)

    expected = {
        1: 0.3,
        2: 0.5,
        3: 0.7,
    }

    for document_id, expected_score in expected.items():

        assert math.isclose(
            fused_scores[document_id],
            expected_score,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )

    print("\n✓ Score fusion test passed")

    callable_scores = fusion(scores)

    for document_id, expected_score in expected.items():

        assert math.isclose(
            callable_scores[document_id],
            expected_score,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )

    print("✓ Callable interface test passed")

    identical_scores = {
        1: 5.0,
        2: 5.0,
        3: 5.0,
    }

    normalized = fusion.normalize(
        identical_scores
    )

    assert normalized == {
        1: 0.0,
        2: 0.0,
        3: 0.0,
    }

    print("✓ Equal-score normalization test passed")

    assert fusion.normalize({}) == {}

    assert fusion.fuse({}) == {}

    print("✓ Empty-score test passed")

from ranking.reranker import Reranker


def test_reranker() -> None:

    scores = {
        1: 0.72,
        2: 0.91,
        3: 0.83,
        4: 0.91,
    }

    reranker = Reranker()

    results = reranker.rerank(
        scores,
        top_k=3,
    )

    print("\nReranker test")
    print("Input scores:")
    print(scores)

    print("Ranked results:")
    print(results)

    assert results == [
        (2, 0.91),
        (4, 0.91),
        (3, 0.83),
    ]

    print("Reranker test passed.")
    

    from ranking.reranker import Reranker


def test_reranker() -> None:

    scores = {
        1: 0.72,
        2: 0.91,
        3: 0.83,
        4: 0.91,
    }

    reranker = Reranker()

    results = reranker.rerank(
        scores,
        top_k=3,
    )

    print("\n=== RERANKER TEST ===")

    print("\nInput scores:")
    print(scores)

    print("\nRanked results:")
    print(results)

    expected = [
        (2, 0.91),
        (4, 0.91),
        (3, 0.83),
    ]

    assert results == expected

    print("\n✓ Reranking test passed")

    callable_results = reranker(
        scores,
        top_k=2,
    )

    expected_callable_results = [
        (2, 0.91),
        (4, 0.91),
    ]

    assert callable_results == expected_callable_results

    print("✓ Callable interface test passed")

    all_results = reranker.rerank(
        scores
    )

    expected_all_results = [
        (2, 0.91),
        (4, 0.91),
        (3, 0.83),
        (1, 0.72),
    ]

    assert all_results == expected_all_results

    print("✓ Full ranking test passed")

    empty_results = reranker.rerank(
        {}
    )

    assert empty_results == []

    print("✓ Empty-score test passed")

    zero_results = reranker.rerank(
        scores,
        top_k=0,
    )

    assert zero_results == []

    print("✓ Zero Top-K test passed")


if __name__ == "__main__":
    test_score_fusion()
    test_reranker()

    print("All tests passed.")