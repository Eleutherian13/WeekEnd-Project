
class StopWordRemover:

    def __init__(self) :
        self.stopwords = STOPWORDS

    def remove(self , tokens : list[str]) -> list[str] :

        return [token for token in tokens if token not in self.stopwords]

    def __call__(self , tokens : list[str]) -> list[str] :
        return self.remove(tokens)

    def __repr__(self) -> str :
        return f"{self.__class__.__name__}()"


STOPWORDS = {"the" , "is" , "in" , "and" , "to" , "a" , "of" , "that" , "it" , "on" , "for" , "with" , "as" , "was" , "at" , "by" , "an" , "be" , "this" , "which"}

if __name__ == "__main__" :
    stopword_remover = StopWordRemover()
    print(stopword_remover.remove(["machine" , "learning" , "ai" , "the" , "is" , "in" , "and" , "to" , "a" , "of" , "that" , "it" , "on" , "for" , "with" , "as" , "was" , "at" , "by" , "an" , "be" , "this" , "which"]))
    

