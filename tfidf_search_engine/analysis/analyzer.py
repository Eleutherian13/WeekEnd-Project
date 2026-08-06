from analysis.regex_tokenizer import RegexTokenizer
from analysis.normalizer import Normalizer


class Analyzer:
    """
    Complete text analysis pipeline.
    """

    def __init__(self):
        self.tokenizer = RegexTokenizer()
        self.normalizer = Normalizer()

    def analyze(self, text: str) -> list[str]:
        tokens = self.tokenizer(text)
        tokens = self.normalizer(tokens)
        return tokens

    def __call__(self, text: str) -> list[str]:
        return self.analyze(text)