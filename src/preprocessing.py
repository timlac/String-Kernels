#! /usr/bin/env python

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import numpy as np
import re
import os
from utils import shuffle_lists


def data(train,
         test,
         texts,
         classes,
         n_train_samples=10000,
         n_test_samples=4000,
         filter_classes=[]):
    """
    Processes the reuters data to lists of text and classes.

    Reads the Reuters corpus from files,
    filters the data by the documents classes, shuffles it and then returns
    n_train_samples and n_test_samples of data.

    Parameters
    ----------
    n_train_samples : int
        Number of training samples to be returned.
    n_test_samples : int
        Number of test samples to be returned.
    filter_classes : list(str)
        A list of the classes that should be included in the dataset.

    Returns
    -------
    train_texts : list(str) [n_train_samples]
        List of the document texts of the train subset.
    train_classes : list(str) [n_train_samples]
        List of the classes for each document of the train subset.
    test_texts : list(str) [n_test_samples]
        List of the document texts of the test subset.
    test_classes : list(str) [n_train_samples]
        List of the classes for each document of the test subset.
    """

    train = set(train)
    test = set(test)

    train_texts = []
    train_classes = []

    test_texts = []
    test_classes = []

    for id, text in texts.items():
        if id in train:
            train_texts.append(text)
            train_classes.append(classes[id])
        elif id in test:
            test_texts.append(text)
            test_classes.append(classes[id])

    if filter_classes:
        train_texts, train_classes = filter_by_class(
            train_texts, train_classes, filter_classes)

        test_texts, test_classes = filter_by_class(test_texts, test_classes,
                                                   filter_classes)

    train_texts, train_classes, test_texts, test_classes = shuffle_lists(
        train_texts, train_classes, test_texts, test_classes)

    train_texts = train_texts[:n_train_samples]
    train_classes = train_classes[:n_train_samples]

    test_texts = test_texts[:n_test_samples]
    test_classes = test_classes[:n_test_samples]

    return train_texts, train_classes, test_texts, test_classes


def filter_by_class(texts, classes, filter_classes):
    """
    Removes all documents which classes are not in the filter_classes parameter.

    Parameters
    ----------
    texts : list(str)
        List of document texts.
    classes : list(list(str))
        List of classes for each document.

    Returns
    -------
    new_text_list : list(str)
        The filtered document list.
    new_text_list : list(list(str))
        The filtered document classes list.

    """
    new_text_list = []
    new_classes_list = []
    for i, classes in enumerate(classes):
        filtered_classes = []
        for cl in classes:
            if cl in filter_classes:
                filtered_classes.append(cl)
        if filtered_classes:
            new_text_list.append(texts[i])
            new_classes_list.append(filtered_classes)
    return new_text_list, new_classes_list


def preprocess_regex(text):
    """
    Filters a string.

    Decapitalizes, removes stopwords and all punctuation from the string.

    Parameters
    ----------
    text : string
    Returns
    -------
    string
        The preprocessed string.

    """
    p1 = re.compile(r'([^a-zA-Z\s\'])')  # Removes everything except a-Z, and '
    p2 = re.compile(r'(\'(\w+)\')')  # Remove quotes
    text = p2.sub(r'\g<2>', p1.sub(r'', text).lower())
    words = text.split()
    word_filter = set(stopwords.words('english'))
    filtered_words = [
        word.replace('\'', '') for word in words if word not in word_filter
    ]
    return ' '.join(filtered_words)


def process_directory(path='../data/', category_filter=None):
    """
    Reads and preprocesses the Reuters-21578 dataset.

    The dataset is split using the "ModApte"-split.

    Parameters
    ----------
    path : string (optional)
        Path to dataset directory
    category_filter : list(str) (optional)
        A list of which categories the documents should have.

    Returns
    -------
    train : list(int)
        List of document-IDs which have the parameter "TRAIN".
    test : list(int)
        List of document-IDS which have the parameter "TEST".
    titles : dict(int : str)
        Map of document-IDS and the titles of the documents.
    texts : dict(int : str)
        Map of document-IDS and the texts of the documents.
    classes: dict(int : list(str))
        Map of document-IDS and the class or classes of the documents.
    """
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
    """
    Reads and preprocesses a file of the Reuters-21578 dataset.

    The dataset is split using the "ModApte"-split.

    Parameters
    ----------
    filename : string (optional)
        Filename of the sgml file.
    category_filter : list(str) (optional)
        A list of which categories the documents should have.

    Returns
    -------
    train : list(int)
        List of document-IDs which have the parameter "TRAIN".
    test : list(int)
        List of document-IDS which have the parameter "TEST".
    titles : dict(int : str)
        Map of document-IDS and the titles of the documents.
    texts : dict(int : str)
        Map of document-IDS and the texts of the documents.
    classes: dict(int : list(str))
        Map of document-IDS and the class or classes of the documents.
    """
    train = []
    test = []
    titles = {}
    texts = {}
    classes = {}

    with open(filename, 'r') as sgml_file:
        corpus = BeautifulSoup(sgml_file.read(), 'html.parser')

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
                    text = title + ' ' + body
                    texts[document_id] = preprocess_regex(text)

                    if document['lewissplit'] == 'TRAIN':
                        train.append(document_id)
                    else:
                        test.append(document_id)

    return train, test, titles, texts, classes


def process_file_2(filename, category_filter=None):
    """
    Reads and preprocesses a file of the Reuters-21578 dataset.

    The dataset is split using the "ModApte"-split.

    Parameters
    ----------
    filename : string (optional)
        Filename of the sgml file.
    category_filter : list(str) (optional)
        A list of which target the documents should have.

    Returns
    -------
    train : list(int)
        List of document-IDs which have the parameter "TRAIN".
    test : list(int)
        List of document-IDS which have the parameter "TEST".
    titles : dict(int : str)
        Map of document-IDS and the titles of the documents.
    texts : dict(int : str)
        Map of document-IDS and the texts of the documents.
    classes: dict(int : list(str))
        Map of document-IDS and the class or classes of the documents.
    """
    train_index_id_map = {}
    train_data = []
    train_target = []

    test_index_id_map = {}
    test_data = []
    test_target = []

    with open(filename, 'r') as sgml_file:
        corpus = BeautifulSoup(sgml_file.read(), 'html.parser')

    for document in corpus('reuters'):

        # Check if document is "ModApte"
        # According to the README (VIII.B.)
        # Training: lewissplit=train, topics=yes
        # Testing: lewissplit=test, topics=yes
        if document['topics'] == 'YES' and (document['lewissplit'] == 'TRAIN'
                                            or
                                            document['lewissplit'] == 'TEST'):
            document_id = int(document['newid'])
            target = []
            data = ''

            for topic in document.topics.contents:
                if category_filter is not None:
                    if any(category in topic.contents
                           for category in category_filter):
                        target.extend(topic.contents)
                else:
                    target.extend(topic.contents)
            if target:
                if document.title is None:
                    title = ''
                else:
                    title = document.title.contents[0]

            if document.body is None:
                body = ''
            else:
                body = document.body.contents[0]

            data = title + ' ' + body
            data = preprocess_regex(data)

            if document['lewissplit'] == 'TRAIN':
                train_index_id_map[len(train_data)] = document_id
                train_data.append(data)
                train_target.append(target)
            else:
                test_index_id_map[len(test_data)] = document_id
                test_data.append(data)
                test_target.append(target)
    data = {}
    test = {}
    train = {}
    test['map'] = test_index_id_map
    test['data'] = test_data
    test['target'] = test_target
    train['map'] = train_index_id_map
    train['data'] = train_data
    train['target'] = train_target
    data['test'] = test
    data['train'] = train
    return data


def get_all_classes(filename='../data/all-topics-strings.lc.txt'):
    """
    Get all classes from the Reuters-21578 dataset.

    Parameters
    ----------
    filename : str
        Filename of file containing a text file where each row is a unique class.

    Returns
    -------
    list(str)
        List of classes

    """
    with open(filename, 'r') as label_file:
        labels = label_file.read().split('\n')
    return labels


def get_classes(classes,
                document_index,
                filename='../data/all-topics-strings.lc.txt',
                category_filter=None):
    """
    Get classes from the Reuters-21578 dataset as a binary vector for each document.

    Parameters
    ----------
    classes : dict(int : list(str))
        Dict that maps the document ID to the list of classes of that document.
    document_index : dict(int : int)
        Dict that maps the document ID to the matrix row index used.
    category_filter : list(str) (optional)
        List of classes which classes will be used.

    Returns
    -------
    label_index : dict(int : str)
        Dict that maps the matrix row index of the class matrix to the label.
    y : array of shape [n_documents, n_classes]
        The class labels of each document as a binary row vector.
    """
    if category_filter is not None:
        labels = category_filter
    else:
        labels = get_all_classes(filename)
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
    # Reverse dict
    label_index = {v: k for k, v in label_index.items()}
    return label_index, y
