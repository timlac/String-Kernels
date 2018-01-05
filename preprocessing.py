from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import os


# Should all numbers be removed as well?
# Should a advanced tokenizer be used or should we keep it simple?
def preprocess_nltk(text):
    words = word_tokenize(text.lower())
    word_filter = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in word_filter]
    return ' '.join(filtered_words)


def preprocess_regex(text):
    p1 = re.compile(r'([^\w\s\'])')  # Removes everything except a-Z, and '.
    p2 = re.compile(r'(\'(\w+)\')')  # Remove quotes
    text = p2.sub(r'\g<2>', p1.sub(r'', text).lower())
    words = text.split()
    word_filter = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in word_filter]
    return ' '.join(filtered_words)


def process_directory(path):
    filenames = [
        filename for filename in os.listdir(path)
        if filename.startswith('reut2-')
    ]

    train = []
    test = []
    titles = {}
    texts = {}
    categories = {}

    for filename in filenames:
        print(filename)
        _train, _test, _titles, _texts, _categories = process_file(
            path + filename)
        train.extend(_train)
        test.extend(_test)
        titles.update(_titles)
        texts.update(_texts)
        categories.update(_categories)

    return train, test, titles, texts, categories


def process_file(filename):
    # lists with ids for which documents are in training and testing
    train = []
    test = []

    # data
    titles = {}
    texts = {}
    categories = {}

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
                categories[document_id] = []

                for topic in document.topics.contents:
                    categories[document_id].append(' '.join(topic.contents))

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

    return train, test, titles, texts, categories
