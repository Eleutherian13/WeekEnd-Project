class Query:
    """
    Represents a user's raw search query.
    """

    def __init__(self, text: str) -> None:

        if not isinstance(text, str):
            raise TypeError(
                "text must be a string"
            )

        if not text.strip():
            raise ValueError(
                "query text cannot be empty"
            )

        self.text = text

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"({self.text!r})"
        )