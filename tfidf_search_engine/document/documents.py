from dataclasses import dataclass , field
from typing import Any


@dataclass(slots=True)
class Document : 

    document_id : int 
    text : str 
    metadata : dict[str , Any] = field(default_factory = dict)


    def __post_init__(self) -> None : 

        if not isinstance(self.text , str ) : 
            raise TypeError("text must be a string")

        if not isinstance(self.document_id , int ) :
            raise TypeError("document_id must be an integer")

        if self.document_id < 0 : 
            raise ValueError("document_id must be a positive integer")

    def __len__(self) -> int : 

        return len(self.text)


    def __repr__(self) -> str  :

        return f"{self.__class__.__name__}({self.document_id!r}, {self.text!r})"



