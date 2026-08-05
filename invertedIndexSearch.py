documents = {
    1: "apple banana",
    2: "banana orange",
    3: "apple orange apple"
}

# this is the documents that we have let's say
# we are implementing a simple inverted index search machine here --> for building the understanding
# we are going to use a dictionary to store the inverted index


inverted_index = {}

for doc_id , content in documents.items() :
    words = content.lower().split()

    for word in words : 
        if word in inverted_index : 
            inverted_index[word].add(doc_id)
        else : 
            inverted_index[word] = set()

            inverted_index[word].add(doc_id)



print(inverted_index)

query = "apple"

results = inverted_index.get(query, [])

print(results)




# this is the output we are searching for 

# {
#  'apple': [1, 3],
#  'banana': [1, 2],
#  'orange': [2, 3]
# }

# query = "apple AND orange"
def inverted_index_and(query) :
    terms = query.lower().split(" and ")

    print(terms)

    result = None 

    left = set(inverted_index.get(terms[0] , []))
    right = set(inverted_index.get(terms[1] , []))

    if left and right : 
        result = left & right 

    return result 

print("This is the output we are searching for exisits in the document number : ")
print(inverted_index_and("apple AND orange"))



def inverted_index_or(query) :

    terms = query.lower().split(" or ")

    result = None 

    left = set(inverted_index.get(terms[0] , []))
    right = set(inverted_index.get(terms[1] , []))

    if left and right : 
        result = left | right 

    return result 

print("This is the output we are searching for exisits in the document number : ")
print(inverted_index_or("apple OR orange"))



