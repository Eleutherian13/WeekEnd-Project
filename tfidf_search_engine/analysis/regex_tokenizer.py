import re

from analysis.tokenizer import Tokenizer


class RegexTokenizer(Tokenizer):
    """
    Tokenizes using regex.
    Removes punctuation.
    """

    def tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text)