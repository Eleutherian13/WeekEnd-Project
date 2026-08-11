from collections.abc import Iterator 
from .documents import Document 

class Corpus :

    def __init__(self ) -> None : 
        self._documents : dict[int , Document] = {}



    def add(self , document: Document) -> None : 

        if not isinstance(document , Document ) : 
            raise TypeError("document must be a Document")

        if document.document_id in self._documents : 
            raise ValueError("document_id must be unique")

        if not isinstance(document.document_id , int ) : 
            raise TypeError("document_id must be an integer")

        self._documents[document.document_id] = document


    def get(self , document_id : int) -> Document | None : 

        try : 
            return self._documents[document_id]

        except KeyError : 
            raise KeyError(f"document_id {document_id} not found or does not exist in the corpus ")

    def remove(self, document_id : int ) -> None : 

        try : 
            return self._documents.pop(document_id)

        except KeyError : 
            raise KeyError(f"document_id {document_id} not found or does not exist in the corpus ")


    def __contains__(self , document_id : int ) -> bool : 
        return document_id in self._documents


    def __getitem__(self , document_id : int ) -> Document : 
        return self.get(document_id)

    def __setitem__(self , document_id : int , document : Document ) -> None :
        self.add(document)

    def __iter__(self ) -> Iterator[Document] : 
        return iter(self._documents.values())

    def __len__(self ) -> int :
        return len(self._documents)

    def __repr__(self) -> str :
        return f"{self.__class__.__name__}()"


    # this is enough for the corpus.py 



    