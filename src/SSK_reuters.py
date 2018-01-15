import random
from math import sqrt
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import time

from preprocessing import get_classes, process_directory, data
from utils import *
from SSK import kernel
from tfidf import train_target
from postprocessing import evaluate


class GramCalc:
    """class to hold information and calculate Gram Matrix efficiently"""

    stored_normalization = None
    counter = 0

    def __init__(self, S, T, n, kernel, symmetric=True):
        self.n = n
        self.kernel = kernel

        self.S = S
        self.T = T

        self.mat = np.zeros((len(S), len(T)))
        self.normalized_mat = np.zeros((len(S), len(T)))

        self.train_normalization = np.zeros(len(S))
        self.test_normalization = np.zeros(len(S))

        self.symmetric = symmetric

    @classmethod
    def store_normalization_vars(cls, vars):
        """store computed normalization values"""
        cls.stored_normalization = vars

    def get_stored_normalization(cls):
        return cls.stored_normalization

    def calculate(self, parallel=True):
        """perform all calculations"""

        if parallel:
            print("building matrix parallel")
            start = time.time()
            self.build_mat_parallel()
            end = time.time()
            print("\ndone with matrix")
            print(self.mat)
            print('\nelapsed time: ', end - start)
        else:
            print("building matrix regular")
            start = time.time()
            self.build_mat()
            end = time.time()
            print("\ndone with matrix")
            print(self.mat)
            print('\nelapsed time: ', end - start)


        if self.symmetric:
            self.train_normalization = self.mat.diagonal()
            self.store_normalization_vars(self.train_normalization)
        else:
            self.train_normalization = self.get_stored_normalization()

        self.build_normalized()
        return self.normalized_mat

    def build_mat_parallel(self):
        """precompute kernel on all required combinations"""
        string_combinations = []
        coordinates = []

        for row, s in enumerate(self.S):
            for col, t in enumerate(self.T):
                if self.symmetric and row > col:
                    pass
                else:
                    string_combinations.append([s, t])
                    coordinates.append([row, col])

        pool = Pool(4)
        outputs = pool.map(self.redirect_to_kernel, string_combinations)
        pool.close()
        pool.join()

        for i in range(len(outputs)):
            c = coordinates[i]
            self.mat[c[0], c[1]] = outputs[i]

        if self.symmetric:
            self.mat = self.symmetrize(self.mat)

    def redirect_to_kernel(self, sc):
        return kernel(sc[0], sc[1], self.n)

    def build_mat(self):
        """precompute kernel on all required combinations"""
        for row, s in enumerate(self.S):
            for col, t in enumerate(self.T):

                if self.symmetric and row > col:
                    pass

                else:
                    self.mat[row, col] = self.kernel(s, t, self.n)

        if self.symmetric:
            self.mat = self.symmetrize(self.mat)

    def build_normalized(self):
        """build normalized gram matrix from precomputed kernel values"""
        for row, s in enumerate(self.S):
            for col, t in enumerate(self.T):

                if self.symmetric and row > col:
                    pass

                elif self.symmetric and row == col:
                    self.normalized_mat[row, col] = 1

                else:
                    self.normalized_mat[row, col] = self.normalize(row, col, s)

        if self.symmetric:
            self.normalized_mat = self.symmetrize(self.normalized_mat)

    def normalize(self, row, col, s):
        """normalize gram matrix element"""
        if self.symmetric:
            return self.mat[row, col] / sqrt(self.train_normalization[row] * self.train_normalization[col])

        else:
            if not self.test_normalization[row]:
                self.test_normalization[row] = kernel(s, s, self.n)

            return self.mat[row, col] / sqrt(self.test_normalization[row] * self.train_normalization[col])

    @staticmethod
    def symmetrize(matrix):
        return matrix + matrix.T - np.diag(matrix.diagonal())



def main():
    # n_train_samples = 3
    # n_test_samples = 2
    #
    # # what classes to look at
    # filter_classes = ["ship", "corn"]
    #
    # # train_ids, test_ids, _, texts, classes = process_file('../data/reut2-002.sgm')
    # train_ids, test_ids, _, texts, classes = process_directory()
    #
    # train_texts, train_classes, test_texts, test_classes = data(
    #     train_ids, test_ids, texts, classes, n_train_samples, n_test_samples, filter_classes)
    #
    # # length of subsequences
    # n = 2
    #
    # print(train_texts)
    # print(train_classes)
    #
    # # build Gram matrix
    # GC_train = GramCalc(train_texts, train_texts, n, kernel=kernel)
    # Gram_train_matrix = GC_train.calculate()
    #
    # print(Gram_train_matrix)
    #
    # # make classes into vector and multilabel binarizer, list of classes tranformed to binary vector
    # y_train, mlb = train_target(train_classes, filter_classes)
    # y_test = mlb.transform(test_classes)
    #
    # print(test_texts)
    # print(test_classes)
    #
    # GC_test = GramCalc(test_texts, train_texts, n, kernel=kernel)
    # Gram_test_matrix = GC_test.calculate()
    #
    # classifier = OneVsRestClassifier(SVC(kernel='precomputed'))
    # classifier.fit(Gram_train_matrix, y_train)
    #
    # y_pred = classifier.predict(Gram_test_matrix)
    #
    # print(y_pred)
    # print(mlb.inverse_transform(y_pred))

    n = 2

    train_texts = ['qtly div nine cts vs eight cts prior pay may 12 record march 31 reuter', 'paralax restricted common shares three year warrants buy 318600 restricted shares six dlrs share paralax said holders american video convertible debentures elected exchange paralax restricted common market',
          'sensormatic electronics corp said upped investment checkrobot', 'Also you know this code is guaranteed to be much much, much slower than', 'year warrants buy 318600 restricted shares six dlrs share paralax said holders american video convertible debentures elected exchange paralax restricted']

    # build Gram matrix
    GC_train = GramCalc(train_texts, train_texts, n, kernel=kernel, symmetric=True)
    Gram_train_matrix = GC_train.calculate(parallel=True)
    print("in main")
    print(Gram_train_matrix)


    # test_texts = ['qtly div nine cts vs eight cts prior pay dead may ']
    #
    # GC_test = GramCalc(test_texts, train_texts, n, kernel=kernel, symmetric=False)
    # Gram_test_matrix = GC_test.calculate()


    # evaluate(y_test, y_pred, mlb, filter_classes)


main()
