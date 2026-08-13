
from dataclasses import dataclass


@dataclass(slots=True)
class Posting:
    """
    Represents the occurrence of one term
    in one document.
    """

    document_id: int
    term_frequency: int

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, int):
            raise TypeError("document_id must be an integer")

        if self.document_id < 0:
            raise ValueError("document_id must be a positive integer")

        if not isinstance(self.term_frequency, int):
            raise TypeError("term_frequency must be an integer")

        if self.term_frequency < 0:
            raise ValueError("term_frequency must be a positive integer")

        
if __name__ == "__main__":
    posting = Posting(document_id=1, term_frequency=2)
    print(posting)
    print(posting.document_id)
    print(posting.term_frequency)
