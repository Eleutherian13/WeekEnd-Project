from analysis.tokenizer import Tokenizer


class WhitespaceTokenizer(Tokenizer):
    """
    Tokenizes using whitespace.
    """

    def tokenize(self, text: str) -> list[str]:
        return text.split()