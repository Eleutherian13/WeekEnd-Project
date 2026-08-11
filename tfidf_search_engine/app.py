from analysis.analyzer import Analyzer

from index.builder import IndexBuilder
from index.inverted_index import InvertedIndex
from index.vocabulary import Vocabulary

from query.query_processing import QueryProcessor

from retieval.candidate_generator import CandidateGenerator

from search.search_engine import SearchEngine


def main():

    # --------------------------------------------------
    # 1. Create indexing components
    # --------------------------------------------------

    analyzer = Analyzer()

    vocabulary = Vocabulary()

    inverted_index = InvertedIndex()

    # --------------------------------------------------
    # 2. Build the index
    # --------------------------------------------------

    index_builder = IndexBuilder(
        analyzer,
        vocabulary,
        inverted_index
    )

    documents = {
        1: "machine learning is amazing",

        2: "deep learning powers artificial intelligence",

        3: "machine learning uses neural networks",

        4: "cats chase mice",

        5: "dogs chase cats",

        6: "deep neural networks power AI"
    }

    for document_id, text in documents.items():

        index_builder.add_document(
            document_id,
            text
        )

    # --------------------------------------------------
    # 3. Create query processing
    # --------------------------------------------------

    query_processor = QueryProcessor(
        analyzer
    )

    # --------------------------------------------------
    # 4. Create candidate generator
    # --------------------------------------------------

    candidate_generator = CandidateGenerator(
        index_builder.inverted_index
    )

    # --------------------------------------------------
    # 5. Create search engine
    # --------------------------------------------------

    search_engine = SearchEngine(
        query_processor,
        candidate_generator
    )

    # --------------------------------------------------
    # 6. Search
    # --------------------------------------------------

    print("\n========== SEARCH ==========")

    print(
        "AND:",
        search_engine.search(
            "machine learning",
            mode="AND"
        )
    )

    print(
        "OR:",
        search_engine.search(
            "machine learning",
            mode="OR"
        )
    )


if __name__ == "__main__":
    main()