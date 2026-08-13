class Reranker:
    """
    Reranks candidate documents using their scores
    and returns an ordered result list.
    """

    def _validate_scores(
        self,
        scores: dict[int, float],
    ) -> None:

        if not isinstance(scores, dict):
            raise TypeError(
                "scores must be a dictionary"
            )

        for document_id, score in scores.items():

            if not isinstance(
                document_id,
                int,
            ):
                raise TypeError(
                    "document ids must be integers"
                )

            if isinstance(
                document_id,
                bool,
            ):
                raise TypeError(
                    "document ids must be integers"
                )

            if document_id < 0:
                raise ValueError(
                    "document ids must be non-negative"
                )

            if not isinstance(
                score,
                (int, float),
            ):
                raise TypeError(
                    "scores must contain only numeric values"
                )

            if isinstance(
                score,
                bool,
            ):
                raise TypeError(
                    "scores must contain only numeric values"
                )

    def _validate_top_k(
        self,
        top_k: int | None,
    ) -> None:

        if top_k is None:
            return

        if not isinstance(
            top_k,
            int,
        ):
            raise TypeError(
                "top_k must be an integer or None"
            )

        if isinstance(
            top_k,
            bool,
        ):
            raise TypeError(
                "top_k must be an integer or None"
            )

        if top_k < 0:
            raise ValueError(
                "top_k must be non-negative"
            )

    def rerank(
        self,
        scores: dict[int, float],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:

        self._validate_scores(scores)
        self._validate_top_k(top_k)

        ranked_results = sorted(
            scores.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )

        if top_k is None:
            return ranked_results

        return ranked_results[:top_k]

    def __call__(
        self,
        scores: dict[int, float],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:

        return self.rerank(
            scores,
            top_k,
        )

    def __repr__(self) -> str:

        return (
            f"{self.__class__.__name__}()"
        )