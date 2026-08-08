from index.inverted_index import InvertedIndex 
from index.posting import Posting 
from index.posting_list import PostingList 

index = InvertedIndex()

# this created a new dict of [str , PostingList]

index.add(
    "machine",
    Posting(
        document_id=1,
        term_frequency=2
    )
)

index.add(
    "machine",
    Posting(
        document_id=3,
        term_frequency=1
    )
)

index.add(
    "learning",
    Posting(
        document_id=1,
        term_frequency=1
    )
)

index.add(
    "learning",
    Posting(
        document_id=2,
        term_frequency=1
    )
)

index.add(
    "learning",
    Posting(
        document_id=3,
        term_frequency=1
    )
)