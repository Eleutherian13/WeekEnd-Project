from __future__ import annotations
from abc import ABC, abstractmethod


class DenseRetriever(ABC):

    @abstractmethod
    def retireve(self, query : str , top_k : int | None = None) -> list[tuple[int , float]] : 

        raise NotImplementedError 

    def __call__(self , query : str , top_k : int | None = None) -> list[tuple[int , float]] :

        return self.retireve(query , top_k)

    def __repr__(self) -> str:

        return f"{self.__class__.__name__}()"

    