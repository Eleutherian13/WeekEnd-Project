import spacy

from analysis.tokenizer import Tokenizer


nlp = spacy.load("en_core_web_sm")


class SpacyTokenizer(Tokenizer):
    """
    Tokenizes using spaCy.
    """

    def tokenize(self, text: str) -> list[str]:
        doc = nlp(text)
        return [token.text for token in doc]

if __name__ == "__main__":
    tokenizer = SpacyTokenizer()
    print(tokenizer("This is a test"))