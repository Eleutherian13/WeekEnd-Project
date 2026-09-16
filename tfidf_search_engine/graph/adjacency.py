from __future__ import annotations
from collections import defaultdict
from .edge import Edge 

class AdjacencyList : 

    # maintains graph connectivity by using adjacency list 
    #  source_node -> outgoing edges 


    def __init__(self,) -> None :
        self._outgoing : dict[str , list[Edge]] = defaultdict(list)

    def add(self , edge : Edge ) -> None : 
        self._outgoing[edge.source_node].append(edge)


    def remove_edge(self , edge : Edge) -> None : 

        edges = self._outgoing.get(edge.source_node, [])

        if not edges : 
            return False 

        if edge not in edges : 
            return False 

        try : 
            edges.remove(edge)
        except ValueError : 
            return False 

        return True 


    def neighbors(self, node_id: str) -> tuple[str, ...]:


        # this asks for returning all thenodes who are directly reachable neighboring nodes 

        return tuple(
            edge.target_node
            for edge in self._outgoing.get(node_id, [])
        )

    def outgoing_edges(self , node_id : str ) -> tuple[Edge , ...] : 

        # returns all the outgoing edges from a node 

        return tuple(self._outgoing.get(node_id  , []))

    def has_edge(self , source_node : str , target_node : str ) -> bool : 

        # return whether there is any edge connecting source node to target node 

        return any(edge.target_node  == target_node for edge in self._outgoing.get(source_node , []))

    def has_node(self , node_id : str) -> bool : 

        # returns whether the node exists in a graph or not 

        return node_id in self._outgoing


    def edges(self) -> tuple[Edge , ...] :
        # returns all the edges in the graph 

        return tuple(edge for edges in self._outgoing.values() for edge in edges)

    def remove_node(self , node_id : str) -> None :
        # remove all outoing edges from the node incomdin edges myust be removed seperatel by the owning graoh , becauyse adjacecny is intentionally responsibe only for connectivity 

        self._outgoing.pop(node_id , None)


    def __len__(self) -> int :
        # returns the number of nodes in the graph
        return sum(len(edges ) for edges in self._outgoing.values())

    def __contains__(self , node_id : str) -> bool :
        # returns whether the node is in the graph or not
        return node_id in self._outgoing

    def clear(self) -> None:
        self._outgoing.clear()

    def __repr__(self) -> str :
        return f"AdjacencyList(nodes={len(self)})"


    
