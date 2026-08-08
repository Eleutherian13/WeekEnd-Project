from index.posting_list import PostingList 
from index.posting import Posting 

posting_list = PostingList()

posting_list.add(
    Posting(
        document_id=1,
        term_frequency=2
    )
)

posting_list.add(
    Posting(
        document_id=3,
        term_frequency=1
    )
)

posting_list.add(
    Posting(
        document_id=6,
        term_frequency=4
    )
)
print("Posting List : ")

for posting in posting_list : 
    print(posting)

print("Document Frequency : ")
print(posting_list.document_frequency)

print("Get posting : ")
print(posting_list.get(1))
print(posting_list.get(3))
print(posting_list.get(6))

print("Length : ")
print(len(posting_list))

print("Iterator : ")

print("\nDocument 3:")
print(posting_list.get(3))


print("\nDocument 5:")
print(posting_list.get(5))

