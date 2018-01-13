import random
from math import sqrt
import numpy as np

from preprocessing import get_classes
from utils import *
from SSK_memo import K_memo


class GramCalc:
    """class to hold information and calculate Gram Matrix efficiently"""

    def __init__(self, S, T, n, kernel):
        self.n = n
        self.kernel = kernel
        self.S = S
        self.T = T
        self.mat = np.zeros((len(S), len(T)))
        self.normalized_mat = np.zeros((len(S), len(T)))

    def calculate(self):
        self.build_mat()
        self.build_normalized()
        return self.normalized_mat

    def build_mat(self):
        # precompute kernel on all required combinations
        for row, s in enumerate(self.S):
            for col, t in enumerate(self.T):
                self.mat[row, col] = self.kernel(self.n, s, t)

    def build_normalized(self):
        # build matrix from precomputed kernel values

        for row, s in enumerate(self.S):
            for col, t in enumerate(self.T):

                if row == col:
                    # diagonal
                    self.normalized_mat[row, col] = 1

                elif self.normalized_mat[col, row] != 0:
                    # symmetry
                    self.normalized_mat[row, col] = self.normalized_mat[col, row]

                else:
                    self.normalized_mat[row, col] = self.normalize(row, col)

    def normalize(self, row, col):
        return self.mat[row, col] / sqrt(self.mat[row, row] * self.mat[col, col])


def make_data(index,
              texts,
              classes,
              n_samples,
              n_features,
              category_filter=None):

    random.shuffle(index)
    index = index[0:n_samples]
    texts = filter_dict(index, texts)
    classes = filter_dict(index, classes)

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


def main():
    X = ['qtly div nine cts vs eight cts prior pay may 12 record march 31 reuter', 'paralax restricted common shares three year warrants buy 318600 restricted shares six dlrs share paralax said holders american video convertible debentures elected exchange paralax restricted common market',
         'sensormatic electronics corp said upped investment checkrobot']

    n = 2
    GC = GramCalc(X, X, n, kernel=K_memo)
    Gram_matrix = GC.calculate()

    print(Gram_matrix)


main()