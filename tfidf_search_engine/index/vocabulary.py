class Vocabulary : 

    def __init__(self ) : 
        self._terms = {}

    def add(self , term : str )-> int: 

        if not isinstance(term , str) : 
            raise TypeError("term must be a string")

        if term  in self._terms : 
            return self._terms[term]

        term_id = len(self._terms)

        self._terms[term] = term_id

        return term_id 

    def contains(self , term : str) -> bool : 

        return term in self._terms 


    def get_id(self , term: str) -> int | None : 

        return self._terms[term]


    def __len__(self) -> int : 
        return len(self._terms)


    def __contains__(self , term : str) -> bool :   
        return term in self._terms

    def __getitem__(self , term : str) -> int :
        return self._terms[term]

if __name__ == "__main__" :
    vocabulary = Vocabulary()

    print(vocabulary.add("machine"))
    print(vocabulary.add("learning"))

    print(vocabulary.contains("machine"))
    print(vocabulary.contains("learning"))


    print(vocabulary.get_id("machine"))
    print(vocabulary.get_id("learning"))

    print(vocabulary.__len__())

    print(vocabulary["learning"])


    
    