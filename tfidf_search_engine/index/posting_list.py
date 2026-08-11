from collections.abc import Iterator

from .posting import Posting


class PostingList:
    """
    Collection of postings for a single term.
    """

    def __init__(self) -> None:
        self._postings: list[Posting] = []

    def add(self, posting: Posting) -> None:
        if not isinstance(posting, Posting):
            raise TypeError("posting must be a Posting")

        existing = self.get(posting.document_id)

        if existing is not None:
            raise ValueError(
                f"document_id {posting.document_id} already exists"
            )

        self._postings.append(posting)

    def get(self, document_id: int) -> Posting | None:
        for posting in self._postings:
            if posting.document_id == document_id:
                return posting

        return None

    @property
    def document_frequency(self) -> int:
        return len(self._postings)

    def __iter__(self) -> Iterator[Posting]:
        return iter(self._postings)

    def __len__(self) -> int:
        return len(self._postings)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._postings!r})"


if __name__ == "__main__" :
    posting_list = PostingList()

    posting_list.add(
        Posting(
            document_id=1,
            term_frequency=2
        )
    )

    posting_list.add(
        Posting(
            document_id=3,
            term_frequency=1
        )
    )

    posting_list.add(
        Posting(
            document_id=6,
            term_frequency=4
        )
    )
    print("Posting List : ")

    for posting in posting_list : 
        print(posting)

    print("Document Frequency : ")
    print(posting_list.document_frequency)

    print("Get posting : ")
    print(posting_list.get(1))
    print(posting_list.get(3))
    print(posting_list.get(6))

    print("Length : ")
    print(len(posting_list))

    print("Iterator : ")

    print("\nDocument 3:")
    print(posting_list.get(3))


    print("\nDocument 5:")
    print(posting_list.get(5))


