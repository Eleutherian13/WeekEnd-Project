try:
    from .posting import Posting
    from .posting_list import PostingList
except ImportError:  # pragma: no cover - allows direct script execution
    from posting import Posting
    from posting_list import PostingList

class InvertedIndex:
    def __init__(self):
        self._index: dict[str, PostingList] = {}

    def add(self, term: str, posting: Posting) -> None:

        # these are just the lines for type checks nothing else 
        
        if not isinstance(term , str) : 
            raise TypeError("term must be a string")

        if not isinstance(posting , Posting) : 
            raise TypeError("posting must be a Posting")

        
        if term not in self._index:
            self._index[term] = PostingList()

        self._index[term].add(posting)

    def contain(self, term: str) -> bool:
        return term in self._index

    def get_postings(self, term: str) -> PostingList | None:
        return self._index.get(term)

    def get(self, term: str) -> PostingList | None:
        return self._index.get(term)

    def __contains__(self, term: str) -> bool:
        return term in self._index

    def __getitem__(self, term: str) -> PostingList:
        return self._index[term]

    def __setitem__(self, term: str, postings: PostingList) -> None:
        self._index[term] = postings

    def __len__(self) -> int:
        return len(self._index)

    def __iter__(self):
        return iter(self._index)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._index!r})"

if __name__ == "__main__" :
    index = InvertedIndex()

    # this created a new dict of [str , PostingList]

    index.add(
        "machine",
        Posting(
            document_id=1,
            term_frequency=2
        )
    )

    index.add(
        "machine",
        Posting(
            document_id=3,
            term_frequency=1
        )
    )

    index.add(
        "learning",
        Posting(
            document_id=1,
            term_frequency=1
        )
    )

    index.add(
        "learning",
        Posting(
            document_id=2,
            term_frequency=1
        )
    )

    index.add(
        "learning",
        Posting(
            document_id=3,
            term_frequency=1
        )
    )

    a = index.get_postings("machine")
    print("The document frequency is :",a.document_frequency)
    print("This means the thing work as intended ")

    print("\nMachine postings:")

    machine_postings = index.get_postings("machine")

    for posting in machine_postings:
        print(posting)
    print("\nLearning postings:")

    learning_postings = index.get_postings("learning")

    for posting in learning_postings:
        print(posting)            