class Normalizer:
    """
    Converts every token to lowercase.
    """

    def normalize(self, tokens: list[str]) -> list[str]:
        return [token.lower() for token in tokens]

    def __call__(self, tokens: list[str]) -> list[str]:
        return self.normalize(tokens)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


