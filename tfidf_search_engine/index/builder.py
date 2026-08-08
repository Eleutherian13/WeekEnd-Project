import os
import sys
from collections import Counter

if __package__ in (None, ""):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from analysis.analyzer import Analyzer
from index.posting import Posting
from index.posting_list import PostingList
from index.inverted_index import InvertedIndex
from index.vocabulary import Vocabulary
from index.forward_index import ForwardIndex

class IndexBuilder:
    def __init__(self, analyzer: Analyzer, vocabulary: Vocabulary, inverted_index: InvertedIndex):
        self.analyzer = analyzer
        self.vocabulary = vocabulary
        self.inverted_index = inverted_index
        self.forward_index = ForwardIndex()

    def add_document(self, document_id: int, text: str):
        tokens = self.analyzer.analyze(text)
        term_counts = Counter(tokens)
        self.forward_index.add_document(
            document_id,
            term_counts
            )

        for term, term_frequency in term_counts.items():
            self.vocabulary.add(term)
            posting = Posting(document_id, term_frequency)

            if term in self.inverted_index:
                self.inverted_index[term].add(posting)
            else:
                self.inverted_index[term] = PostingList()
                self.inverted_index[term].add(posting)


if __name__ == "__main__" :
    analyzer = Analyzer()
    vocabulary = Vocabulary()
    inverted_index = InvertedIndex()

    index_builder = IndexBuilder(analyzer , vocabulary , inverted_index)

    index_builder.add_document(1 , "The cat sat on the mat")
    index_builder.add_document(2 , "The dog chased the cat")
    index_builder.add_document(3 , "The bird flew away")
    index_builder.add_document(4 , "The cat chased the bird")
    index_builder.add_document(5 , "Machine learning is amazing")

    print(index_builder.inverted_index)
    print(index_builder.vocabulary)
    print(index_builder.analyzer)
    print(index_builder.analyzer.analyze("Machine Learning!!"))
    print(index_builder.analyzer.analyze("The Quick Brown Fox"))

    print(index_builder.inverted_index["cat"])
    print(index_builder.inverted_index["bird"])
    print(index_builder.inverted_index["learn"])