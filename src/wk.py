#!/usr/bin/env python
'''Standard word kernel
From the article:
WK is a linear kernel that measures the similarity between documents that are
indexed by words with tfidf weighting scheme.

The entries of the feature vectors are weighted using a variant of tfidf,
log(1+tf) * log(n/df).

tf: term frequency - number of times the term occurs in a document.
df: document frequency - number of documents that contains the term.
n: total number of documents.

The documents are normalised so that each document has equal length.
'''
from collections import Counter
from math import log
from scipy.sparse import lil_matrix
from sklearn.preprocessing import normalize


def create_doc_word_matrix(texts, N):
    corpus = []
    n = len(texts)
    df = Counter()
    doc_map = {}
    for text in texts.values():
        words = text.split(' ')
        corpus.extend(words)
        df.update(words)
    most_common_words = Counter(corpus).most_common(N)
    word_set = set([word for word, _ in most_common_words])
    word_map = {word: index for index, word in enumerate(word_set)}
    mat = lil_matrix((n, N))
    for i, d in enumerate(texts.items()):
        document_id, text = d
        words = text.split(' ')
        tf = Counter([word for word in words if word in word_set])
        for word in tf:
            j = word_map[word]
            mat[i, j] = tfidf(tf[word], df[word], n)
        doc_map[document_id] = i
    return doc_map, normalize(mat.tocsr())


def tfidf(tf, df, n):
    return log(1 + tf) * log(n / df)