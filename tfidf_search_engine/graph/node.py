from __future__ import annotations
from dataclasses import dataclass , field 
from typing import Any 

@dataclass(frozen=True , slots = True)
class Node : 
    # this represents a node in the graph
    # A node has globally unique id , a type/label , optional metadata 


    node_id : str       # gives o(1) average lookup 
    node_type : str         # -- this is optional , but it is good to have a type for the node
    metadata: dict[str , Any] = field(default_factory = dict) 
    # we have not written as metadata : dict[str , Any] | None = None , because we want to avoid the case where metadata is None and we have to check for it every time we access it. Instead we will have an empty dict if there is no metadata.

    #default_factory=dict means:Create a brand-new dictionary for every Node.


    def __post_init__(self) -> None : 
        # why postinit ? need validations after initializartions of classes variabvles 

        if not isinstance(self.node_id , str) : 
            raise TypeError("node_id must be a string")

        if not self.node_id.strip() : 
            raise ValueError("node_id must not be empty")

        if not isinstance(self.node_type , str) : 
            raise TypeError("node_type must be a string")

        if not self.node_type.strip() :
            raise ValueError("node_type must not be empty")

        object.__setattr__(self , "metadata" , dict(self.metadata))
        # now what does this line do ? 

    def get(self , key : str , default: Any = None ) -> Any : 
        return self.metadata.get(key , default)

    def has(self, key : str) -> bool : 
        return key in self.metadata

    