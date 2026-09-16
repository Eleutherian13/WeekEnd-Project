from __future__ import annotations
from dataclasses import dataclass , field 
from typing import Any 

@dataclass(frozen = True , slots = True) 
class Edge : 

    # represents a direct rel between 2 nodes 
    # params ; source node , target node , edge_type : semantic relationship tyupe , metadata  wight optinal 

    source_node : str 
    target_node : str
    edge_type : str 
    weight : float = 1.0
    metadata : dict[str , Any] = field(default_factory=dict)


    def __post_init__(self) -> None : 
        if not isinstance(self.source_node , str) : 
            raise TypeError("source_node must be a string")

        if not self.source_node.strip() :
            raise ValueError("source_node must not be empty")

        if not isinstance(self.target_node , str) :
            raise TypeError("target_node must be a string")

        if not self.target_node.strip() :
            raise ValueError("target_node must not be empty")

        if not isinstance(self.edge_type , str) : 
            raise TypeError("edge_type must be a string")

        if not self.edge_type.strip() :
            raise ValueError("edge_type must not be empty")

        if not isinstance(self.weight , (int , float)) :
            raise TypeError("weight must be a number")

        object.__setattr__(self , "metadata" , dict(self.metadata))

    def get(self, key: str, default: Any = None) -> Any:
        """Return a metadata value."""
        return self.metadata.get(key, default)  




        