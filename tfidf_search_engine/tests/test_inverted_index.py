import unittest

from index.inverted_index import InvertedIndex
from index.posting import Posting


class TestInvertedIndex(unittest.TestCase):
    def test_add_and_retrieve_postings(self):
        index = InvertedIndex()
        index.add("machine", Posting(1, 2))
        index.add("machine", Posting(3, 1))

        postings = index.get_postings("machine")

        self.assertIsNotNone(postings)
        self.assertEqual(
            [(posting.document_id, posting.term_frequency)
             for posting in postings],
            [(1, 2), (3, 1)],
        )
        self.assertTrue(index.contain("machine"))
        self.assertFalse(index.contain("missing"))

    def test_duplicate_document_posting_is_rejected(self):
        index = InvertedIndex()
        index.add("machine", Posting(1, 1))

        with self.assertRaises(ValueError):
            index.add("machine", Posting(1, 2))

    def test_invalid_inputs_are_rejected(self):
        index = InvertedIndex()

        with self.assertRaises(TypeError):
            index.add(1, Posting(1, 1))

        with self.assertRaises(TypeError):
            index.add("machine", "not a posting")


if __name__ == "__main__":
    unittest.main()