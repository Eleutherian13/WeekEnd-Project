# this is the file where we are going to build the inderxer 

from preprocessing import preprocess
from data import documents

index = {}

for doc_id , content in documents.items():

    words = preprocess(content)

    for word in set(words):

        if word not in index :

            index[word] = []

            index[word].append(doc_id)


# this will make the inverted index for the documents we are using 


from pprint import pprint

pprint(index)
# this line just prints the index in a more readable format and this is called as 
# pretty print 




positional_index = {}

for doc_id , content in documents.items():

    words = preprocess(content)

    for position , word in enumerate(words):

        if word not in positional_index :

            positional_index[word] = {}

        if doc_id not in positional_index[word] :

            positional_index[word][doc_id] = []

            positional_index[word][doc_id].append(position)
    

position_index = {}

for doc_id, text in documents.items():

    words = preprocess(text)

    for position, word in enumerate(words):

        if word not in position_index:

            position_index[word] = {}

        if doc_id not in position_index[word]:

            position_index[word][doc_id] = []

        position_index[word][doc_id].append(position)



pprint(position_index["learning"])