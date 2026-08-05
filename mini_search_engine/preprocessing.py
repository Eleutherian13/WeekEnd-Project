import re 

def lower_case(text):
    return text.lower()

def tokenize(text):
    return re.findall(r"\b\w+\b", text)
# what does this function do ? 
# it will tokenize the text and return a list of words
# it will remove the punctuation and return a list of words
# it will remove the stop words and return a list of words
# it will remove the numbers and return a list of words
# it will remove the special characters and return a list of words
 
 
def preprocess(text):
    text = lower_case(text)
    text = tokenize(text)
    return text


# stemming()

# lemmatization()

# stopword removal()

# unicode normalization()


# this would be added without need of changing anything 

# this isthe test for the preprocessing.py file 

# print(preprocess("Machine Learning is AMAZING!!"))