try:
    from index.inverted_index import InvertedIndex
except ImportError:  # pragma: no cover - allows direct script execution
    from ..index.inverted_index import InvertedIndex


class CandidateGenerator: 

    def __init__(self , inverted_index : InvertedIndex) :
        self.inverted_index = inverted_index

    def _get_document_ids(self , term: list[str]) -> set[int] :

        posting_list = self.inverted_index.get(term)

        if posting_list is None : 

            return set()


        return {
            posting.document_id for posting in posting_list 
        } 

    def or_retrive(self , terms : str ) -> set[int] : 

        candidates : set[int] = set()

        for term in terms : 

            candidates.update(self._get_document_ids(term))

        return candidates 

    def and_retrieve(self , terms : list[str]) -> set[int] : 

        if not terms:
            return set()

        candidates = self._get_document_ids(terms[0])

        for term in terms[1: ] : 

            candidates = candidates & self._get_document_ids(term)

        return candidates 


    def __call__(self , terms : list[str] ) -> set[int] :
        return self.and_retrieve(terms)

    def __repr__(self) -> str :
        return f"{self.__class__.__name__}()"





