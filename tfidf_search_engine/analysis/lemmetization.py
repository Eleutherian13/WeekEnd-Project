from nltk.stem import WordNetLemmatizer

class Lemmatizer:
    def __init__(self):
        # Create the lemmatizer once
        self.lemmatizer = WordNetLemmatizer()

    def lemmatize(self, tokens: list[str]) -> list[str]:
        # Apply lemmatization to each token
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def __call__(self, tokens: list[str]) -> list[str]:
        # Allow the object to be called like a function
        return self.lemmatize(tokens)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


if __name__ == "__main__":
    lemmatizer = Lemmatizer()
    tokens = ["running", "runs", "studies", "study", "better", "mice"]
    print(lemmatizer.lemmatize(tokens))
