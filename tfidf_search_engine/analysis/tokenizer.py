# This file should have one responsibility only.

# Convert text into raw words.

# Nothing else.

# Don't lowercase.

# Don't stem.

# Don't remove stopwords.

# Just tokenize.

class Tokenizerv1: 

    def tokenize(self , text : str) -> list[str] :
        return text.split()

    def __call__(self , text : str) -> list[str] : 
        return self.tokenize(text)

    def __repr__(self) -> str :
        return f"{self.__class__.__name__}()"

tokenizer  = Tokenizerv1()

if __name__ == "__main__" :
    print(tokenizer("This is a test"))

text = "This is a test" 



class Tokenizer:
    """
    Base class for all tokenizers.
    """

    def tokenize(self, text: str) -> list[str]:
        raise NotImplementedError(
            "Subclasses must implement tokenize()."
        )

    def __call__(self, text: str) -> list[str]:
        return self.tokenize(text)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
    