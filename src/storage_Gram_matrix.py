import random
from math import sqrt
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from multiprocessing import Pool
from multiprocessing import cpu_count
import time

from preprocessing import get_classes, process_directory, data, process_file
from utils import *
from SSK import kernel
from storage_SSK import KernelOperations
from tfidf import train_target
from postprocessing import evaluate


class GramCalc:
    """class to hold information and calculate Gram Matrix efficiently"""

    stored_normalization = None

    def __init__(self, S, T, N, kernel, symmetric=True):
        self.S = S
        self.T = T

        self.N = N+1
        self.kernel = kernel

        self.mat = np.zeros((self.N, len(S), len(T)))
        self.normalized_mat = np.zeros((self.N, len(S), len(T)))

        self.train_normalization = np.zeros((self.N, len(S)))
        self.test_normalization = np.zeros((self.N, len(S)))

        self.symmetric = symmetric

    @classmethod
    def store_normalization_vars(cls, vars):
        """store computed normalization values"""
        cls.stored_normalization = vars

    def get_stored_normalization(self):
        return self.stored_normalization

    def calculate(self, parallel=True):
        """perform all calculations"""

        if parallel:
            print("building matrix parallel")
            start = time.time()
            self.build_mat_parallel()
            end = time.time()
            print("\ndone with matrix")
            for i in self.mat:
                print("i: ", i)
                print()
            print('\nelapsed time: ', end - start)

        if self.symmetric:
            for n in range(self.N):
                self.train_normalization[n] = self.mat[n].diagonal()
            self.store_normalization_vars(self.train_normalization)

        else:
            self.train_normalization = self.get_stored_normalization()

        self.build_normalized()
        return np.nan_to_num(self.normalized_mat, copy=False)

    def generate_string_combos(self):
        """generate all string combinations required to build gram matrix
        as well as all norm combos"""
        mat_combos = []
        mat_coords = []

        for row, s in enumerate(self.S):
            for col, t in enumerate(self.T):
                if self.symmetric and row > col:
                    pass
                else:
                    mat_combos.append([s, t])
                    mat_coords.append([row, col])

        if not self.symmetric:
            # need to calculate normalization values seperately
            for idx, s in enumerate(self.S):
                mat_combos.append([s, s])
                mat_coords.append([idx, -1])

        # sort according to longest string
        zipped = zip(mat_combos, mat_coords)
        zipped_sorted = sorted(zipped, key=lambda x: len(x[0][0]) * len(x[0][1]), reverse=True)
        separated = list(zip(*zipped_sorted))
        mat_combos = list(separated[0])
        mat_coords = list(separated[1])

        return mat_combos, mat_coords

    def build_mat_parallel(self):
        mat_combos, mat_coords = self.generate_string_combos()

        outputs = self.parallelize(mat_combos)

        print("outputs: ")
        for i in outputs:
            print(i[2])
        print("outputs: ", len(outputs[1]))
        print("N ", self.N)

        for i in range(len(mat_combos)):
            for n in range(self.N):
                print("n ", n )

                c = mat_coords[i]

                # assymetric case
                # normalization values are stored in negative index
                if c[1] < 0:
                    self.test_normalization[n, c[0]] = outputs[i][n]

                else:
                    self.mat[n, c[0], c[1]] = outputs[i][n]

        if self.symmetric:
            for n in range(self.N):
                self.mat[n] = self.symmetrize(self.mat[n])

    def parallelize(self, string_vector):
        pool = Pool(cpu_count())
        outputs = pool.map(self.redirect_to_kernel, string_vector, chunksize=1)
        pool.close()
        pool.join()
        return outputs

    def redirect_to_kernel(self, sc):
        ko = KernelOperations(sc[0], sc[1], self.N)
        ret = ko.run_all_kernels()
        return ret

    def build_normalized(self):
        """build normalized gram matrix from precomputed kernel values"""
        for n in range(self.N):
            for row, s in enumerate(self.S):
                for col, t in enumerate(self.T):

                    if self.symmetric and row > col:
                        pass

                    elif self.symmetric and row == col:
                        self.normalized_mat[n, row, col] = 1

                    else:
                        self.normalized_mat[n, row, col] = self.normalize(n, row, col)

        if self.symmetric:
            for n in range(self.N):
                self.normalized_mat[n] = self.symmetrize(self.normalized_mat[n])

    def normalize(self, n, row, col):
        """normalize gram matrix element"""
        if self.symmetric:
            return self.mat[n, row, col] / sqrt(self.train_normalization[n, row] * self.train_normalization[n, col])

        else:
            return self.mat[n, row, col] / sqrt(self.test_normalization[n, row] * self.train_normalization[n, col])

    @staticmethod
    def symmetrize(matrix):
        return matrix + matrix.T - np.diag(matrix.diagonal())


def main():
    # n_train_samples = 10
    # n_test_samples = 10
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

    # train_texts = ["re", 'oo']

    train_texts =  ['grain reserve holdings breakdown us agriculture department ',
                   'brazil coffee exports disrupted strike dayold strike brazilian seamen affecting coffee'
                   'us grain analysts see lower corn soy planting grain analysts surveyed american',
                   'union shippers agree cut ny port costs new york ',
                   'grain certificate redemptions ']
                   # 'us exporters report tonnes corn switched unknown ussr'
                   # 'midwest cash grain slow country movement cash grain dealers reported slow country movement corn ',
                   # 'brazil seamen continue strike despite court hundreds marines alert key brazilian ports seamen decided remain indefinite strike even higher labour court saturday ruled illegal union leaders said halt first national strike seamen years started february union leaders said would return work unless got pct pay ']

    # build Gram matrix
    GC_train = GramCalc(train_texts, train_texts, n, kernel=kernel, symmetric=True)
    Gram_train_matrix = GC_train.calculate(parallel=True)
    print("in main")
    print("Gram train matrix")
    for i in Gram_train_matrix:
        print(i)
        print()

    print("\n")

    test_texts = ['grain certificate redemptions put mln']

    GC_test = GramCalc(test_texts, train_texts, n, kernel=kernel, symmetric=False)
    Gram_test_matrix = GC_test.calculate(parallel=True)

    print("Gram test matrix")
    for i in Gram_test_matrix:
        print(i)
        print()


    # evaluate(y_test, y_pred, mlb, filter_classes)
def get_train_texts(n_samples):
    _, _, _, texts, _ = process_file('../data/reut2-000.sgm')
    train_texts = list(texts.values())[:n_samples]
    return sorted(train_texts, key=len, reverse=True)


def test(train_texts):
    n = 2
    GC_train = GramCalc(
        train_texts, train_texts, n, kernel=kernel, symmetric=True)
    Gram_train_matrix = GC_train.calculate(parallel=True)
    return Gram_train_matrix


if __name__ == '__main__':
    main()
