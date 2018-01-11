from preprocessing import process_file, get_classes
from SVM import build_matrix
import random
from utils import split_data
from wk import create_doc_word_matrix

def make_data(index,
              texts,
              classes,
              n_samples,
              n_features,
              category_filter=None):

    random.shuffle(index)
    index = index[0:n_samples]
    texts = split_data(index, texts)
    classes = split_data(index, classes)

    document_index = {}
    X = []
    mapper = {}
    for idx, item in enumerate(texts.items()):
        document_id, text = item
        X.append(text)
        mapper[idx] = document_id
        document_index[document_id] = idx

    label_index, y = get_classes(classes, document_index, category_filter=category_filter)

    return document_index, label_index, X, y, mapper


# params
categories = ['earn', 'acq', 'ship', 'corn']
n_train_samples = 5000
n_test_samples = 100
n_features = 3000
n_samples = 2

# read all data
train_index, test_index, titles, texts, classes = process_file('../data/reut2-003.sgm', categories)

document_index, label_index, X, y, mapper = make_data(
    train_index, texts, classes, n_samples, n_features, categories)

print(X)
print(len(X))

n = 2
print("buiding matrix....")
Gram = build_matrix(n, X, X)

print(Gram)

print(label_index)
print(y)