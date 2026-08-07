from typing import Iterator
from posting import Posting 


class PostingList : 

    def __init__(self) :

        self._posting : list[Posting] = []

    def add (self , posting : Posting ) -> None: 

        self._posting.append(posting)


    def get(self , document_id : int) -> Posting | None : 

        for posting in self._posting : 

            if posting.document_id == document_id :
                 return posting 
        return None 

    @property 
    def document_frequency(self) -> int : 

        return len(self._posting)

    def __iter__(self) -> Iterator[Posting] : 

        return iter(self._posting)

    def __len__(self) -> int : 

        return len(self._posting)

if __name__ == "__main__" :
    posting_list = PostingList()

    posting_list.add(Posting(document_id=1, term_frequency=2))
    posting_list.add(Posting(document_id=2, term_frequency=3))

    print(posting_list.get(1))
    print(posting_list.get(2))

    print(posting_list.document_frequency)

    for posting in posting_list :
        print(posting)