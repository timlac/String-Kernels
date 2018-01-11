#! /usr/bin/env python

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import re
import os


def preprocess_nltk(text):
    words = word_tokenize(text.lower())
    word_filter = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in word_filter]
    return ' '.join(filtered_words)


def preprocess_regex(text):
    p1 = re.compile(r'([^\w\s\'])')  # Removes everything except a-Z, and '
    p2 = re.compile(r'(\'(\w+)\')')  # Remove quotes
    text = p2.sub(r'\g<2>', p1.sub(r'', text).lower())
    words = text.split()
    word_filter = set(stopwords.words('english'))
    filtered_words = [
        word.replace('\'', '') for word in words if word not in word_filter
    ]
    return ' '.join(filtered_words)


def process_directory(path='../data/', category_filter=None):
    filenames = [
        filename for filename in os.listdir(path)
        if filename.startswith('reut2-')
    ]

    train = []
    test = []
    titles = {}
    texts = {}
    classes = {}

    for filename in filenames:
        print(filename)
        _train, _test, _titles, _texts, _classes = process_file(
            path + filename, category_filter)
        train.extend(_train)
        test.extend(_test)
        titles.update(_titles)
        texts.update(_texts)
        classes.update(_classes)

    return train, test, titles, texts, classes


def process_file(filename, category_filter=None):
    # lists with ids for which documents are in training and testing
    train = []
    test = []

    # data
    titles = {}
    texts = {}
    classes = {}

    with open(filename, 'r') as sgml_file:
        corpus = BeautifulSoup(sgml_file.read())

        for document in corpus('reuters'):
            # Check if document is "ModApte"
            # According to the README (VIII.B.)
            # Training: lewissplit=train, topics=yes
            # Testing: lewissplit=test, topics=yes
            if document['topics'] == 'YES' and (
                    document['lewissplit'] == 'TRAIN'
                    or document['lewissplit'] == 'TEST'):
                document_id = int(document['newid'])
                categories = []

                for topic in document.topics.contents:
                    if category_filter is not None:
                        if any(category in topic.contents
                               for category in category_filter):
                            categories.extend(topic.contents)
                    else:
                        categories.extend(topic.contents)
                if categories:
                    classes[document_id] = categories
                    if document.title is None:
                        title = ''
                    else:
                        title = document.title.contents[0]

                    titles[document_id] = title

                    if document.body is None:
                        body = ''
                    else:
                        body = document.body.contents[0]
                    texts[document_id] = preprocess_regex(body)

                    if document['lewissplit'] == 'TRAIN':
                        train.append(document_id)
                    else:
                        test.append(document_id)

    return train, test, titles, texts, classes


def get_all_labels(filename='../data/all-topics-strings.lc.txt'):
    with open(filename, 'r') as label_file:
        labels = label_file.read().split('\n')
    return labels


def get_labels(classes,
               document_index,
               filename='../data/all-topics-strings.lc.txt',
               category_filter=None):
    if category_filter is not None:
        labels = category_filter
    else:
        labels = get_all_labels(filename)
    label_index = {label: index for index, label in enumerate(labels)}
    y = np.zeros((len(classes), len(label_index)))
    label_filter = set(labels)
    for document_id, document_labels in classes.items():
        if document_labels is not None:
            i = document_index[document_id]
            for label in document_labels:
                if label in label_filter:
                    j = label_index[label]
                    y[i, j] = 1
    return label_index, y
