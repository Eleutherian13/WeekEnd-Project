class QueryExpander:
    """
    Expands already-processed query terms using
    a user-provided term -> related terms mapping.
    """

    def __init__(
        self,
        expansions: dict[str, list[str]] | None = None,
    ) -> None:

        if expansions is None:
            expansions = {}

        if not isinstance(expansions, dict):
            raise TypeError(
                "expansions must be a dictionary"
            )

        for term, related_terms in expansions.items():

            if not isinstance(term, str):
                raise TypeError(
                    "expansion terms must be strings"
                )

            if not isinstance(related_terms, list):
                raise TypeError(
                    "expanded terms must be lists"
                )

            if not all(
                isinstance(item, str)
                for item in related_terms
            ):
                raise TypeError(
                    "expanded terms must contain only strings"
                )

        self.expansions = expansions

    def expand(
        self,
        terms: list[str],
    ) -> list[str]:

        if not isinstance(terms, list):
            raise TypeError(
                "terms must be a list"
            )

        if not all(
            isinstance(term, str)
            
            for term in terms
        ):
            raise TypeError(
                "terms must contain only strings"
            )

        expanded_terms = []

        for term in terms:

            # Always preserve original term.
            if term not in expanded_terms:
                expanded_terms.append(term)

            # Add expansion terms.
            for expanded_term in self.expansions.get(
                term,
                [],
            ):

                if expanded_term not in expanded_terms:
                    expanded_terms.append(
                        expanded_term
                    )

        return expanded_terms

    def __call__(
        self,
        terms: list[str],
    ) -> list[str]:

        return self.expand(terms)

    def __repr__(self) -> str:

        return (
            f"{self.__class__.__name__}("
            f"expansions={self.expansions!r}"
            f")"
        )