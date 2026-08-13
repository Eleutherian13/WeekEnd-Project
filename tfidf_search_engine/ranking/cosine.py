import math


class CosineSimilarity:

    def _validate_vector(
        self,
        vector: list[float],
        name: str,
    ) -> None:

        if not isinstance(vector, list):
            raise TypeError(
                f"{name} must be a list"
            )

        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            for value in vector
        ):
            raise TypeError(
                f"{name} must contain only numeric values"
            )

    def dot_product(
        self,
        vector_a: list[float],
        vector_b: list[float],
    ) -> float:

        self._validate_vector(
            vector_a,
            "vector_a",
        )

        self._validate_vector(
            vector_b,
            "vector_b",
        )

        if len(vector_a) != len(vector_b):

            raise ValueError(
                "vectors must have the same length"
            )

        total = 0.0

        for i in range(len(vector_a)):

            total = (
                total
                + vector_a[i] * vector_b[i]
            )

        return total

    def norm(
        self,
        vector: list[float],
    ) -> float:

        self._validate_vector(
            vector,
            "vector",
        )

        total = 0.0

        for value in vector:

            total = (
                total
                + value ** 2
            )

        return math.sqrt(total)

    def similarity(
        self,
        vector_a: list[float],
        vector_b: list[float],
    ) -> float:

        self._validate_vector(
            vector_a,
            "vector_a",
        )

        self._validate_vector(
            vector_b,
            "vector_b",
        )

        if len(vector_a) != len(vector_b):

            raise ValueError(
                "vectors must have the same length"
            )

        dot_product = self.dot_product(
            vector_a,
            vector_b,
        )

        norm_a = self.norm(vector_a)
        norm_b = self.norm(vector_b)

        if norm_a == 0.0 or norm_b == 0.0:

            return 0.0

        return (
            dot_product
            / (norm_a * norm_b)
        )

    def __call__(
        self,
        vector_a: list[float],
        vector_b: list[float],
    ) -> float:

        return self.similarity(
            vector_a,
            vector_b,
        )

    def __repr__(self) -> str:

        return (
            f"{self.__class__.__name__}()"
        )


    