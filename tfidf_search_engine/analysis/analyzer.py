from analysis.regex_tokenizer import RegexTokenizer
from analysis.stemmer import Stemmer
from analysis.stopwords import StopWordRemover
from analysis.normalizer import Normalizer
from analysis.lemmetization import Lemmatizer


class Analyzer:

    def __init__(
        self,
        tokenizer=None,
        normalizer=None,
        stopword_remover=None,
        stemmer=None,
        use_stemmer: bool = True,
    ):
        self.tokenizer = tokenizer or RegexTokenizer()
        self.normalizer = normalizer or Normalizer()
        self.stopword_remover = stopword_remover or StopWordRemover()
        self.processor = stemmer if stemmer is not None else (
            Stemmer() if use_stemmer else Lemmatizer()
        )

    def analyze(self, text: str) -> list[str]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        tokens = self.tokenizer(text) if callable(self.tokenizer) else self.tokenizer.tokenize(text)
        tokens = self.normalizer(tokens) if callable(self.normalizer) else self.normalizer.normalize(tokens)
        tokens = self.stopword_remover(tokens) if callable(self.stopword_remover) else self.stopword_remover.remove(tokens)

        if isinstance(self.processor, Stemmer):
            tokens = self.processor.stem(tokens)
        else:
            tokens = self.processor.lemmatize(tokens)

        return tokens

    def __call__(self, text: str):
        return self.analyze(text)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

# if __name__ == "__main__" :
#     analyzer = Analyzer()
#     print(analyzer.analyze("Machine Learning!!"))
#     print(analyzer.analyze("The Quick Brown Fox"))



# this approach to run the file is wrong as the it does not know about the package analysis 
# instead for runniing get in the file app.py and write a script for running it 
