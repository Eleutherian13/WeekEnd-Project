from analysis.stemmer import Stemmer


class ForwardIndex : 

    def __init__(self ) : 

        self._document : dict[int , dict[str , int ]] = {}


    def add_document(self, document_id : int , term_frequency : dict[str , int]) -> None: 

        if not isinstance(document_id , int ) : 
            raise TypeError("document_id must be an integer")

        if document_id < 0 :
            raise ValueError("document_id must be a positive integer")

        if not isinstance(term_frequency , dict) :
            raise TypeError("term_frequency must be a dictionary")

        if not all(isinstance(key , str) for key in term_frequency.keys()) :
            raise TypeError("term_frequency keys must be strings")

        self._document[document_id] = dict(term_frequency)


    def get_document(self , document_id : int) -> dict[str , int] | None : 

        return self._document.get(document_id)

    def get_document_length(self, document_id: int) -> int:
        document = self.get_document(document_id)
        if document is None:
            return 0
        return sum(document.values())

    def get_term_frequency(self, document_id : int , term : str) -> int | None : 

        document = self._document.get(document_id)

        if document is None : 
            return 0 

        value = document.get(term, None)
        if value is not None:
            return value

        stemmed_term = Stemmer().stem([term])[0]
        if stemmed_term == term:
            return 0

        return document.get(stemmed_term, 0)

    def contains(self , document_id : int ) -> bool : 
        return document_id in self._document

    def __len__(self ) -> int : 

        return len(self._document)

    def __repr__(self) -> str :
        return f"{self.__class__.__name__}()"

    def document_ids(self) : 
        return self._document.keys()

if __name__ == "__main__" :
    forward_index = ForwardIndex()


    forward_index.add_document(
        1,
        {
            "machine": 2,
            "learning": 1,
            "neural": 3
        }
    )

    forward_index.add_document(
        2,
        {
            "deep": 1,
            "learning": 2
        }
    )


    print("Number of documents:")
    print(len(forward_index))


    print("\nDocument 1:")
    print(forward_index.get_document(1))


    print("\nMachine TF in Doc 1:")
    print(
        forward_index.get_term_frequency(
            1,
            "machine"
        )
    )


    print("\nLearning TF in Doc 1:")
    print(
        forward_index.get_term_frequency(
            1,
            "learning"
        )
    )


    print("\nBanana TF in Doc 1:")
    print(
        forward_index.get_term_frequency(
            1,
            "banana"
        )
    )

