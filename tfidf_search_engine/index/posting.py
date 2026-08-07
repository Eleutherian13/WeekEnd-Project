
from dataclasses import dataclass


@dataclass(slots=True)
class Posting:
    """
    Represents the occurrence of one term
    in one document.
    """

    document_id: int
    term_frequency: int


if __name__ == "__main__":
    posting = Posting(document_id=1, term_frequency=2)
    print(posting)
    print(posting.document_id)
    print(posting.term_frequency)
