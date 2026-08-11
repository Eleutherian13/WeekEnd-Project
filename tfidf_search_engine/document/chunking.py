from dataclasses import dataclass , field
from typing import Any

@dataclass(slots=True)
class Chunk : 

    chunk_id : int 
    text : str 
    metadata : dict[str , Any ] = field(default_factory = dict)
    position : int = 0

    def __post_init__(self) -> None : 

        if not isinstance(self.text , str) : 
            raise TypeError("text must be a string")

        if not isinstance(self.chunk_id , int ) :
            raise TypeError("chunk_id must be an integer")

        if self.chunk_id < 0 :
            raise ValueError("chunk_id must be a positive integer")

        if not isinstance(self.metadata , dict) :
            raise TypeError("metadata must be a dictionary")

        if not all(isinstance(key , str) for key in self.metadata.keys()) :
            raise TypeError("metadata keys must be strings")

        if not isinstance(self.position , int ) : 
            raise TypeError("position must be an integer")

        if self.position < 0 :
            raise ValueError("position must be a positive integer")

    def __len__(self) -> int :
        return len(self.text)

    def __repr__(self) -> str :
        return f"{self.__class__.__name__}({self.chunk_id!r}, {self.text!r})"

    def __eq__(self , other : object ) -> bool :

        if not isinstance(other , Chunk ) : 
            return False

        return self.chunk_id == other.chunk_id



        