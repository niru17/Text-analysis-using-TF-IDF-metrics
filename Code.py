import os
import math
import collections
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Set the root directory for the corpus
corpusroot = "./US_Inaugural_Addresses"
token = RegexpTokenizer(r'[a-zA-Z]+')  # Regular expression-based tokenizer for words

# Function to read a document from a file
def read_doc(f):
    f = open(os.path.join(corpusroot, f), "r", encoding='utf-8')  # Open file in the corpus
    document = f.read()  # Read the opened file
    f.close()  # Close the file
    document = document.lower()  # Convert the document to lowercase
    return document

# Function to read all documents in the corpus
def read_docs():
    docs = []
    for f in os.listdir(corpusroot):
        if f.startswith('0') or f.startswith('1') or f.startswith('2') or f.startswith('3'):
            doc = read_doc(f)
            docs.append((f, doc))
    return docs

# Set of English stopwords
stop_words = set(stopwords.words('english'))

# Initialize Porter Stemmer for stemming
stem = PorterStemmer()

# List to store tokenized documents along with their frequency terms
tokenized_doc_list = []

# Function to tokenize a document, filter stopwords, and stem tokens
def token_doc(doc):
    tokens = token.tokenize(doc)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    stemmed_tokens = [stem.stem(token) for token in filtered_tokens]
    return collections.Counter(stemmed_tokens)

# Function to tokenize all documents in the corpus
def token_docs(docs):
    token_docs = []
    for doc in docs:
        f = doc[0]
        s_tokens = token_doc(doc[1])
        f_terms = collections.Counter(s_tokens)
        token_docs.append(f_terms)
        tokenized_doc_list.append((f, f_terms))
    return token_docs

# Function to preprocess a token (convert to lowercase and stem)
def pre_process_token(t):
    t = t.lower()
    t = stem.stem(t)
    return t

# Function to calculate the inverse document frequency (IDF) for a given token
def find_idf(token, token_docs):
    token = pre_process_token(token)
    no_token_docs = sum(1 for doc_tokens in token_docs if token in doc_tokens)
    if no_token_docs == 0:
        return 0
    else:
        return math.log10(len(token_docs) / no_token_docs)

# Function to calculate the term frequency (TF) for a given token in a document
def find_tf(token, tokens):
    token = pre_process_token(token)
    token_count = tokens[token]
    if token_count == 0:
        return 0
    else:
        return 1 + math.log10(token_count)

# Function to calculate the normalized weight of a token in a document
def getidf(token):
    filtered_docs = []
    token = pre_process_token(token)
    if len(tokenized_doc_list) == 0:
        filtered_docs = token_docs(read_docs())
        idf = find_idf(token, filtered_docs)
    else:
        list_tokens = [item[1] for item in tokenized_doc_list]
        idf = find_idf(token, list_tokens)
    return -1 if idf <= 0 else idf

# Function to calculate the normalized weight of a token in a document
def norm(doc):
    sum_w = sum((find_tf(token, doc) * getidf(token)) ** 2 for token in doc)
    return math.sqrt(sum_w)

# Function to perform a query and find the most similar document
def query(string):
    q_tokens = token_doc(string)
    q_vector = [find_tf(token, q_tokens) / norm(q_tokens) for token in q_tokens]
    most_similardocument = ""
    docs = read_docs()
    max_similar = -1
    for doc in docs:
        doc_tokens = token_doc(doc[0])
        similar = sum(getweight(doc[0], token) * q_vector[index] for index, token in enumerate(q_tokens))
        if similar > max_similar:
            max_similar = similar
            most_similardocument = doc[0]

    return (most_similardocument, max_similar)

# Function to calculate the normalized weight of a token in a document
def getweight(f, token):
    idf = getidf(token)
    doc_tokens = [item[1] for item in tokenized_doc_list if item[0] == f]
    tf = find_tf(token, doc_tokens[0])
    w = tf * idf / norm(doc_tokens[0])
    return w

# Call the function and inspect the output
# result = query("your_query_here")
# print(result)

# Print IDF and weights for specific tokens and documents
print("%.12f" % getidf('children'))
print("%.12f" % getidf('foreign'))
print("%.12f" % getidf('people'))
print("%.12f" % getidf('honor'))
print("%.12f" % getidf('great'))
print("--------------")
print("%.12f" % getweight('19_lincoln_1861.txt','constitution'))
print("%.12f" % getweight('23_hayes_1877.txt','public'))
print("%.12f" % getweight('25_cleveland_1885.txt','citizen'))
print("%.12f" % getweight('09_monroe_1821.txt','revenue'))
print("%.12f" % getweight('05_jefferson_1805.txt','press'))
print("--------------")
print("(%s, %.12f)" % query("pleasing people"))
print("(%s, %.12f)" % query("war offenses"))
print("(%s, %.12f)" % query("british war"))
print("(%s, %.12f)" % query("texas government"))
print("(%s, %.12f)" % query("cuba government"))
