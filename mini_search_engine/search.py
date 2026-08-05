from indexer import index 

def search(query) :

    query = query.lower()

    results = index.get(query , [])

    return results 

def and_search(query1 , query2):

    left = set(index.get(query1 , []))
    right = set(index.get(query2 , []))

    return sorted(list(left & right))

print(and_search("cat" , "dog"))
print(and_search("cat" , "bird"))

def or_search(query1 , query2):

    left = set(index.get(query1 , []))
    right = set(index.get(query2 , []))

    return sorted(list(left | right))


print(or_search("cat" , "bird"))
print(or_search("cat" , "dog"))

