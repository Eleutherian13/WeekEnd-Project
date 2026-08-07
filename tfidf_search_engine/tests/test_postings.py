from index.posting import Posting


def test_posting() : 

    posting = Posting(document_id = 1 , term_frequency = 2) 

    assert posting.document_id == 1
    assert posting.term_frequency == 2

    assert repr(posting) == "Posting(document_id=1, term_frequency=2)"

    assert str(posting) == "Posting(document_id=1, term_frequency=2)"


    print(posting)

    print(posting.document_id)
    print(posting.term_frequency)

test_posting()

