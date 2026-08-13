from analysis.stemmer import Stemmer
from index.forward_index import ForwardIndex
from index.inverted_index import InvertedIndex


class Statistics:
    """
    Provides corpus-level statistics required by
    retrieval and ranking algorithms.
    """

    def __init__(
        self,
        forward_index: ForwardIndex,
        inverted_index: InvertedIndex
    ):
        self.forward_index = forward_index
        self.inverted_index = inverted_index

    def document_count(self) -> int:
        """
        Return the total number of indexed documents.
        """
        return len(self.forward_index)

    def document_length(self, document_id: int) -> int:
        """
        Return the number of analyzed terms in a document.
        """
        return self.forward_index.get_document_length(document_id)

    def total_terms(self) -> int:
        """
        Return the total number of analyzed terms
        across the entire corpus.
        """
        total = 0

        for document_id in self.forward_index.document_ids():
            total += self.document_length(document_id)

        return total

    def average_document_length(self) -> float:
        """
        Return the average analyzed document length.
        """
        count = self.document_count()

        if count == 0:
            return 0.0

        return self.total_terms() / count

    def document_frequency(self, term: str) -> int:
        """
        Return the number of documents containing a term.
        """

        posting_list = self.inverted_index.get(term)
        if posting_list is not None:
            return len(posting_list)

        stemmed_term = Stemmer().stem([term])[0]
        if stemmed_term == term:
            return 0

        posting_list = self.inverted_index.get(stemmed_term)
        if posting_list is None:
            return 0

        return len(posting_list)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


if __name__ == "__main__":

    from analysis.analyzer import Analyzer
    from index.builder import IndexBuilder

    documents = {
        1: "machine learning is amazing",
        2: "deep learning powers artificial intelligence",
        3: "machine learning uses neural networks",
        4: "cats chase mice",
        5: "dogs chase cats",
        6: "deep neural networks power AI"
    }

    analyzer = Analyzer()

    builder = IndexBuilder(analyzer)

    for document_id, text in documents.items():
        builder.add_document(document_id, text)

    stats = builder.statistics

    print("Documents:", stats.document_count())

    print(
        "Doc 1 length:",
        stats.document_length(1)
    )

    print(
        "Doc 3 length:",
        stats.document_length(3)
    )

    print(
        "Total terms:",
        stats.total_terms()
    )

    print(
        "Average document length:",
        stats.average_document_length()
    )

    print(
        "DF(machine):",
        stats.document_frequency("machine")
    )

    print(
        "DF(learning):",
        stats.document_frequency("learning")
    )

    print(
        "DF(cats):",
        stats.document_frequency("cats")
    )