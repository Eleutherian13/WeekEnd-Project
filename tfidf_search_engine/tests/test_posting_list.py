import unittest

from index.posting import Posting
from index.posting_list import PostingList


class TestPostingList(unittest.TestCase):
    def test_add_get_iteration_and_frequency(self):
        postings = PostingList()
        postings.add(Posting(1, 2))
        postings.add(Posting(3, 1))

        self.assertEqual(len(postings), 2)
        self.assertEqual(postings.document_frequency, 2)
        self.assertEqual(postings.get(1), Posting(1, 2))
        self.assertIsNone(postings.get(9))
        self.assertEqual(
            [posting.document_id for posting in postings],
            [1, 3],
        )

    def test_duplicate_document_ids_are_rejected(self):
        postings = PostingList()
        postings.add(Posting(1, 1))

        with self.assertRaises(ValueError):
            postings.add(Posting(1, 2))

    def test_non_posting_values_are_rejected(self):
        with self.assertRaises(TypeError):
            PostingList().add("posting")


if __name__ == "__main__":
    unittest.main()

