documents = {
    1: "apple iphone ai",
    2: "apple pie recipe",
    3: "google ai model",
    4: "microsoft copilot ai"
}

# this is the documents that we have let's say 
# we are implementing a simple boolean search machine here --> for building the understanding 


def boolean_and(query1 , query2) :
    result = []

    for doc_id , content in documents.items() :
        words = set(content.split())

        if query1 in words and query2 in words :
            result.append(doc_id)

    return result 

print(boolean_and("apple" , "ai"))


def boolean_or(query1 , query2) :
    result = []

    for doc_id , content in documents.items():
        words = set(content.split())

        if query1 in words or query2 in words : 
            result.append(doc_id)

    return result 


print(boolean_or("apple" , "ai"))


def boolean_not(query) :
    result = []

    for doc_id , content in documents.items():
        words = set(content.split())

        if query not in words : 
            result.append(doc_id)

    return result

print(boolean_not("apple"))

def boolean_search(query) :
    result = []

    for doc_id , content in documents.items():
        words = set(content.split())

        if query in words : 
            result.append(doc_id)

    return result

print(boolean_search("ai"))
print("kun faya kun ")



