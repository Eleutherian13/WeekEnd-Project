from __future__ import annotations 
import math 
from collections.abc import Sequence


Number = int | float 

class VectorSimilarity : 

    @staticmethod 
    def _validate_vector(vector : Sequence[Number] , name : str = "vector") -> list[float] : 

        if not isinstance(vector , Sequence) :
            raise TypeError(f"{name} must be a sequence of numbers.")

        if len(vector) == 0 :
            raise ValueError(f"{name} must be non-empty.")

        values : list[float] = []

        for value in vector : 

            if isinstance(value , bool) : 
                raise TypeError(f"{name} must be a sequence of numbers.")

            if not isinstance(value , (int , float)) : 
                raise TypeError(f"{name} must be a sequence of numbers.")

            values.append(float(value))

        return values 


    @staticmethod
    def _validate_same_dimention(vector1 : Sequence[Number] , vector2 : Sequence[Number]) -> None : 

        if len(vector1) != len(vector2) : 
            raise ValueError("Vectors must have the same dimension.")

    @classmethod 
    def dot_product(cls , vector1 : Sequence[Number] , vector2 : Sequence[Number]) -> float : 
        
        a = cls._validate_vector(vector1 , name = "vector1")
        b = cls._validate_vector(vector2 , name = "vector2")

        cls._validate_same_dimention(a , b)

        return sum(a[i] * b[i] for i in range(len(a)))

    @classmethod 
    def cosine_similarity(cls , vector1 : Sequence[Number] , vector2: Sequence[Number]) -> float :

        a = cls._validate_vector(vector1 , name = "vector1")
        b = cls._validate_vector(vector2 , name = "vector2")

        cls._validate_same_dimention(a , b)

        magnitude_a = math.sqrt(sum(a[i] ** 2 for i in range(len(a))))
        magnitude_b = math.sqrt(sum(b[i] ** 2 for i in range(len(b))))

        if magnitude_a == 0.0:
            raise ValueError(
                "Cannot calculate cosine similarity for a zero vector."
            )

        if magnitude_b == 0.0:
            raise ValueError(
                "Cannot calculate cosine similarity for a zero vector."
            )

        return cls.dot_product(vector1 , vector2) / (magnitude_a * magnitude_b)

    @classmethod 
    def euclidean_distance(cls , vector1 : Sequence[Number] , vector2 : Sequence[Number]) -> float :

        a = cls._validate_vector(vector1 , name = "vector1")
        b = cls._validate_vector(vector2 , name = "vector2")

        cls._validate_same_dimention(a , b)

        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


    
    
