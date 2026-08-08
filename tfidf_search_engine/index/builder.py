import os
import sys
from collections import Counter

if __package__ in (None, ""):
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )

    if project_root not in sys.path:
        sys.path.insert(0, project_root)


from analysis.analyzer import Analyzer

from index.posting import Posting
from index.posting_list import PostingList
from index.inverted_index import InvertedIndex
from index.vocabulary import Vocabulary
from index.forward_index import ForwardIndex
from index.positional_index import PositionalIndex
from index.statistics import Statistics


class IndexBuilder:

    def __init__(
        self,
        analyzer: Analyzer,
        vocabulary: Vocabulary,
        inverted_index: InvertedIndex
    ):
        self.analyzer = analyzer
        self.vocabulary = vocabulary
        self.inverted_index = inverted_index

        # Stores document -> term frequencies
        self.forward_index = ForwardIndex()

        # Stores term -> document -> positions
        self.positional_index = PositionalIndex()

        # Corpus-level statistics
        self.statistics = Statistics(
            self.forward_index,
            self.inverted_index
        )

    def add_document(
        self,
        document_id: int,
        text: str
    ) -> None:

        # --------------------------------------------------
        # 1. Analyze the document
        # --------------------------------------------------

        tokens = self.analyzer.analyze(text)

        # --------------------------------------------------
        # 2. Build term frequencies
        # --------------------------------------------------

        term_counts = Counter(tokens)

        # --------------------------------------------------
        # 3. Add document to Forward Index
        # --------------------------------------------------

        self.forward_index.add_document(
            document_id,
            term_counts
        )

        # --------------------------------------------------
        # 4. Add positional information
        # --------------------------------------------------

        for position, term in enumerate(tokens):

            self.positional_index.add(
                term,
                document_id,
                position
            )

        # --------------------------------------------------
        # 5. Build Inverted Index + Vocabulary
        # --------------------------------------------------

        for term, term_frequency in term_counts.items():

            # Add term to vocabulary
            self.vocabulary.add(term)

            # Create posting for this document
            posting = Posting(
                document_id,
                term_frequency
            )

            # Add posting to existing posting list
            if term in self.inverted_index:

                self.inverted_index[term].add(posting)

            # Create a new posting list
            else:

                self.inverted_index[term] = PostingList()

                self.inverted_index[term].add(posting)


if __name__ == "__main__":

    analyzer = Analyzer()
    vocabulary = Vocabulary()
    inverted_index = InvertedIndex()

    index_builder = IndexBuilder(
        analyzer,
        vocabulary,
        inverted_index
    )

    # --------------------------------------------------
    # Test documents
    # --------------------------------------------------

    documents = {
        1: "machine learning is amazing",

        2: "deep learning powers artificial intelligence",

        3: "machine learning uses neural networks",

        4: "cats chase mice",

        5: "dogs chase cats",

        6: "deep neural networks power AI"
    }

    # --------------------------------------------------
    # Build the complete index
    # --------------------------------------------------

    for document_id, text in documents.items():

        index_builder.add_document(
            document_id,
            text
        )

    # --------------------------------------------------
    # Basic information
    # --------------------------------------------------

    print("\n========== ANALYZER ==========")

    print(
        index_builder.analyzer.analyze(
            "Machine Learning!!"
        )
    )

    # --------------------------------------------------
    # Vocabulary
    # --------------------------------------------------

    print("\n========== VOCABULARY ==========")

    print(index_builder.vocabulary)

    # --------------------------------------------------
    # Inverted Index
    # --------------------------------------------------

    print("\n========== INVERTED INDEX ==========")

    print(index_builder.inverted_index)

    # --------------------------------------------------
    # Positional Index
    # --------------------------------------------------

    print("\n========== POSITIONAL INDEX ==========")

    stemmed_machine = index_builder.analyzer.analyze("machine")[0]
    stemmed_learning = index_builder.analyzer.analyze("learning")[0]

    print(
        "machine in Doc 1:",
        index_builder.positional_index.get_positions(
            stemmed_machine,
            1
        )
    )

    print(
        "learning in Doc 1:",
        index_builder.positional_index.get_positions(
            stemmed_learning,
            1
        )
    )

    print(
        "machine in Doc 3:",
        index_builder.positional_index.get_positions(
            stemmed_machine,
            3
        )
    )

    print(
        "learning in Doc 3:",
        index_builder.positional_index.get_positions(
            stemmed_learning,
            3
        )
    )

    # --------------------------------------------------
    # Forward Index
    # --------------------------------------------------

    print("\n========== FORWARD INDEX ==========")

    print(
        "Doc 1 length:",
        index_builder.forward_index.get_document_length(1)
    )

    print(
        "Doc 3 length:",
        index_builder.forward_index.get_document_length(3)
    )

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    print("\n========== STATISTICS ==========")

    print(
        "Document count:",
        index_builder.statistics.document_count()
    )

    print(
        "Total terms:",
        index_builder.statistics.total_terms()
    )

    print(
        "Average document length:",
        index_builder.statistics.average_document_length()
    )

    print(
        "DF(machine):",
        index_builder.statistics.document_frequency(
            stemmed_machine
        )
    )

    print(
        "DF(learning):",
        index_builder.statistics.document_frequency(
            stemmed_learning
        )
    )

    print(
        "DF(cats):",
        index_builder.statistics.document_frequency(
            index_builder.analyzer.analyze("cats")[0]
        )
    )