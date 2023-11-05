#this code implements a simple search engine.
#The code reads a corpus and produces TF-IDF vectors for documents in the corpus.
#Then, given a query string, the code returns the query answer - the document with the highest cosine similarity score for the query.

import os
import math
from nltk.tokenize import RegexpTokenizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter

corpusroot = 'put your folder path here'
#list to store the tokens from the corpus
corpus_tokens = []
#tokenizer to tokenize the tokens
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
#using porter stemmer
stemmer = PorterStemmer()

for filename in os.listdir(corpusroot):
    if filename.startswith('0') or filename.startswith('1'):
        file = open(os.path.join(corpusroot, filename), "r", encoding='windows-1252')
        doc = file.read()
        file.close()
        doc = doc.lower()
        tokens = tokenizer.tokenize(doc)
        corpus_tokens.append(tokens)

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

#function to remnove the stopwords from the tokens list
def remove_stopwords(tokens):
    return [token for token in tokens if token not in stop_words]

corpus_tokens_no_stopwords = [remove_stopwords(doc_tokens) for doc_tokens in corpus_tokens]

#function to stem tokens
def stem_tokens(tokens):
    return [stemmer.stem(token) for token in tokens]

corpus_tokens_stemmed = [stem_tokens(doc_tokens) for doc_tokens in corpus_tokens_no_stopwords]

#function to caluclate log base 10 which will be used in the further calculations
def log10(x):
    return math.log10(x) if x > 0 else 0

#function to calculate the idf weights for the tokens in the documents
def getidf(token):
    token = stemmer.stem(token)
    doc_count = sum(1 for doc_tokens in corpus_tokens_stemmed if token in doc_tokens)
    if doc_count > 0:
        return math.log10(len(corpus_tokens_stemmed) / doc_count)
    else:
        return -1

#this is to calulate the idf weights of the tokens that have not been stemmed
def getidf2(token):
    doc_count = sum(1 for doc_tokens in corpus_tokens_stemmed if token in doc_tokens)
    if doc_count > 0:
        return math.log10(len(corpus_tokens_stemmed) / doc_count)
    else:
        return -1

doc_filenames = [filename for filename in os.listdir(corpusroot) if filename.startswith('0') or filename.startswith('1')]
doc_tokens_dict = {filename: tokens for filename, tokens in zip(doc_filenames, corpus_tokens_stemmed)}

def get_tokens_by_filename(filename):
    return doc_tokens_dict.get(filename, [])

#function to calulcate the tf.idf wieghts from the documents
def getweight(filename, token):
    stemmed_token = stemmer.stem(token)
    tokens_in_doc = get_tokens_by_filename(filename)
    term_count = tokens_in_doc.count(stemmed_token)

    if term_count == 0:
        return 0

    term_tf = 1 + log10(term_count)
    term_idf = getidf(token)
    raw_weight_token = term_tf * term_idf

    #this is used to calculate raw weights for all tokens in the document
    raw_weights_list = []
    for unique_token in set(tokens_in_doc):
        tf = 1 + log10(tokens_in_doc.count(unique_token))
        #here we are using the second idf function, so that we dont end up doing stemming double times
        idf = getidf2(unique_token)
        raw_weights_list.append(tf * idf)

    #this will calculate magnitude to do the normalization
    raw_squares = sum([weight**2 for weight in raw_weights_list])
    magnitude = math.sqrt(raw_squares)

    #this will compute the normalized weight
    normalized_weight = raw_weight_token / magnitude if magnitude > 0 else 0
    return normalized_weight

#this is the function to get the tf.idf weitghs for the query based on "lnc" scheme
def query(qstring):
    q_tokens = tokenizer.tokenize(qstring.lower())
    q_tokens_stemmed = stem_tokens(remove_stopwords(q_tokens))

    raw_weights_q = {}
    for token in q_tokens_stemmed:
        tf_q = 1 + log10(q_tokens_stemmed.count(token))
        idf_q = 1
        raw_weights_q[token] = tf_q * idf_q

    magnitude_q = math.sqrt(sum([weight**2 for weight in raw_weights_q.values()]))

    normalized_weights_q = {}
    if magnitude_q > 0:
        for token, weight in raw_weights_q.items():
            normalized_weights_q[token] = weight / magnitude_q

    max_similarity = 0
    max_sim_doc_filename = None

    for doc_filename, doc_tokens in doc_tokens_dict.items():
        doc_similarity = sum(normalized_weights_q.get(term, 0) * getweight(doc_filename, term) for term in normalized_weights_q)

        if doc_similarity > max_similarity:
            max_similarity = doc_similarity
            max_sim_doc_filename = doc_filename

    return (max_sim_doc_filename, max_similarity)

print("%.12f" % getidf('british'))
print("%.12f" % getidf('union'))
print("%.12f" % getidf('war'))
print("%.12f" % getidf('military'))
print("%.12f" % getidf('great'))
print("--------------")
print("%.12f" % getweight('02_washington_1793.txt','arrive'))
print("%.12f" % getweight('07_madison_1813.txt','war'))
print("%.12f" % getweight('12_jackson_1833.txt','union'))
print("%.12f" % getweight('09_monroe_1821.txt','british'))
print("%.12f" % getweight('05_jefferson_1805.txt','public'))
print("--------------")
print("(%s, %.12f)" % query("pleasing people"))
print("(%s, %.12f)" % query("british war"))
print("(%s, %.12f)" % query("false public"))
print("(%s, %.12f)" % query("people institutions"))
print("(%s, %.12f)" % query("violated willingly"))
