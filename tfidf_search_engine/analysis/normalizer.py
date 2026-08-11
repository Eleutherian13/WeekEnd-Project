class Normalizer:
    """
    Converts every token to lowercase.
    """

    def normalize(self, tokens: list[str]) -> list[str]:

        if not isinstance(tokens, list):
            raise TypeError("tokens must be a list")

        # this is the list actual line that is responsible for converting the tokens to lowercase
        return [token.lower() for token in tokens]

    def __call__(self, tokens: list[str]) -> list[str]:
        return self.normalize(tokens)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


