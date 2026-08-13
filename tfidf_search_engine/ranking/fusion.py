class ScoreFusion:
    """
    Combines multiple document-ranking score signals
    using weighted score fusion.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
    ) -> None:

        if weights is None:
            weights = {}

        self._validate_weights(weights)

        self.weights = weights.copy()

    def _validate_weights(
        self,
        weights: dict[str, float],
    ) -> None:

        if not isinstance(weights, dict):
            raise TypeError(
                "weights must be a dictionary"
            )

        for name, weight in weights.items():

            if not isinstance(name, str):
                raise TypeError(
                    "weight names must be strings"
                )

            if not isinstance(
                weight,
                (int, float),
            ):
                raise TypeError(
                    "weights must contain only numeric values"
                )

            if isinstance(weight, bool):
                raise TypeError(
                    "weights must contain only numeric values"
                )

            if weight < 0:
                raise ValueError(
                    "weights must be non-negative"
                )

    def _validate_scores(
        self,
        scores: dict[str, dict[int, float]],
    ) -> None:

        if not isinstance(scores, dict):
            raise TypeError(
                "scores must be a dictionary"
            )

        for name, document_scores in scores.items():

            if not isinstance(name, str):
                raise TypeError(
                    "score names must be strings"
                )

            if not isinstance(
                document_scores,
                dict,
            ):
                raise TypeError(
                    "document scores must be dictionaries"
                )

            for document_id, score in (
                document_scores.items()
            ):

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

                if not isinstance(
                    score,
                    (int, float),
                ):
                    raise TypeError(
                        "scores must contain only numeric values"
                    )

                if isinstance(score, bool):
                    raise TypeError(
                        "scores must contain only numeric values"
                    )

    def normalize(
        self,
        scores: dict[int, float],
    ) -> dict[int, float]:

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

            if not isinstance(
                score,
                (int, float),
            ):
                raise TypeError(
                    "scores must contain only numeric values"
                )

            if isinstance(score, bool):
                raise TypeError(
                    "scores must contain only numeric values"
                )

        if not scores:
            return {}

        minimum = min(scores.values())
        maximum = max(scores.values())

        if minimum == maximum:
            return {
                document_id: 0.0
                for document_id in scores
            }

        return {
            document_id: (
                score - minimum
            ) / (
                maximum - minimum
            )
            for document_id, score in scores.items()
        }

    def fuse(
        self,
        scores: dict[str, dict[int, float]],
    ) -> dict[int, float]:

        self._validate_scores(scores)

        if not scores:
            return {}

        combined_scores: dict[int, float] = {}

        for signal_name, document_scores in (
            scores.items()
        ):

            weight = self.weights.get(
                signal_name,
                0.0,
            )

            if weight == 0.0:
                continue

            normalized_scores = self.normalize(
                document_scores
            )

            for document_id, score in (
                normalized_scores.items()
            ):

                combined_scores[document_id] = (
                    combined_scores.get(
                        document_id,
                        0.0,
                    )
                    + weight * score
                )

        return combined_scores

    def __call__(
        self,
        scores: dict[str, dict[int, float]],
    ) -> dict[int, float]:

        return self.fuse(scores)

    def __repr__(self) -> str:

        return (
            f"{self.__class__.__name__}("
            f"weights={self.weights!r}"
            f")"
        )