import unittest

from index.posting import Posting


class TestPosting(unittest.TestCase):
    def test_valid_posting_preserves_fields_and_type(self):
        posting = Posting(document_id=1, term_frequency=2)

        self.assertIsInstance(posting, Posting)
        self.assertEqual(posting.document_id, 1)
        self.assertEqual(posting.term_frequency, 2)

    def test_invalid_posting_values_are_rejected(self):
        with self.assertRaises(TypeError):
            Posting(document_id="1", term_frequency=2)

        with self.assertRaises(ValueError):
            Posting(document_id=1, term_frequency=-1)

        self.assertEqual(
            Posting(document_id=1, term_frequency=True).term_frequency,
            True,
        )


if __name__ == "__main__":
    unittest.main()

