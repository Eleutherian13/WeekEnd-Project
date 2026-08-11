from nltk.stem import PorterStemmer

class Stemmer:
    def __init__(self):
        # Create the stemmer once
        self.stemmer = PorterStemmer()

    def stem(self, tokens: list[str]) -> list[str]:
        # Apply stemming to each token
        return [self.stemmer.stem(token) for token in tokens]


    def process(self , tokens : list[str] ) -> list[str] : 
        return self.stem(tokens)

    def __call__(self, tokens: list[str]) -> list[str]:
        # Allow the object to be called like a function
        return self.stem(tokens)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


if __name__ == "__main__":
    stemmer = Stemmer()
    tokens = ["running", "runs", "studies", "study"]
    print(stemmer.stem(tokens))

