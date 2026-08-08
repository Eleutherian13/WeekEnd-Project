class PositionalIndex:

    def __init__(self):
        self._index: dict[str, dict[int, list[int]]] = {}

    def add(self, term: str, document_id: int, position: int) -> None:
        if term not in self._index:
            self._index[term] = {}

        if document_id not in self._index[term]:
            self._index[term][document_id] = []

        self._index[term][document_id].append(position)

    def get_positions(self, term: str, document_id: int) -> list[int] | None:
        documents = self._index.get(term)
        if documents is None:
            return None

        return documents.get(document_id)

    def contains(self, term: str, document_id: int) -> bool:
        return term in self._index and document_id in self._index[term]

    def get_documents(self, term: str) -> list[int] | None:
        documents = self._index.get(term)
        if documents is None:
            return None
        return list(documents.keys())

    def __contains__(self, term: str) -> bool:
        return term in self._index

    def __len__(self) -> int:
        return len(self._index)

if __name__ == "__main__" :
    index = PositionalIndex()


    index.add("machine", 1, 0)
    index.add("learning", 1, 1)
    index.add("machine", 1, 5)
    index.add("learning", 1, 6)

    index.add("machine", 3, 0)
    index.add("learning", 3, 1)


    print("Positions of machine in Doc 1:")
    print(index.get_positions("machine", 1))


    print("\nPositions of learning in Doc 1:")
    print(index.get_positions("learning", 1))


    print("\nPositions of machine in Doc 3:")
    print(index.get_positions("machine", 3))


    print("\nDocuments containing machine:")
    print(index.get_documents("machine"))


    print("\nDoes machine occur in Doc 1?")
    print(index.contains("machine", 1))


    print("\nDoes machine occur in Doc 5?")
    print(index.contains("machine", 5))


    print("\nNumber of unique terms:")
    print(len(index))