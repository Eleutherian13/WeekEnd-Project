from .tokenizer import Tokenizer 

class WhitespaceTokenizer(Tokenizer) : 

    def tokenize(self , text : str) -> list[str] : 

        if not isinstance(text , str ) :

            raise TypeError("text must be a string")

        return text.split()