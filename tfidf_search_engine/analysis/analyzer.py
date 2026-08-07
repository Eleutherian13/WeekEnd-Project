from analysis.regex_tokenizer import RegexTokenizer 
from analysis.stemmer import Stemmer 
from analysis.stopwords import StopWordRemover
from analysis.whitespace import WhitespaceTokenizer
from analysis.normalizer import Normalizer
from analysis.lemmetization import Lemmatizer


class Analyzer : 


    def __init__(self , use_stemmer : bool = True) :

        self.tokenizer = RegexTokenizer()

        self.normalizer = Normalizer()

        self.stopword_remover = StopWordRemover()

        self.processor = Stemmer() if use_stemmer else Lemmatizer()

    def analyze(self , text : str) -> list[str] : 

        tokens = self.tokenizer(text)

        tokens = self.normalizer(tokens)

        tokens = self.stopword_remover(tokens)

        if isinstance(self.processor , Stemmer ) : 

            tokens = self.processor.stem(tokens)

        else : 
            tokens = self.processor.lemmatize(tokens)

        return tokens

     
    def __call__(self, text: str):
        return self.analyze(text)


    def __repr__(self) -> str :
        return f"{self.__class__.__name__}()"

# if __name__ == "__main__" :
#     analyzer = Analyzer()
#     print(analyzer.analyze("Machine Learning!!"))
#     print(analyzer.analyze("The Quick Brown Fox"))



# this approach to run the file is wrong as the it does not know about the package analysis 
# instead for runniing get in the file app.py and write a script for running it 
