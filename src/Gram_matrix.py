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
from tfidf import train_target
from postprocessing import evaluate


class GramCalc:
    """class to hold information and calculate Gram Matrix efficiently"""

    stored_normalization = None

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
        return np.nan_to_num(self.normalized_mat, copy=False)

    def build_mat_parallel(self):
        mat_combos, mat_coords, norm_combos = self.generate_string_combos()

        string_vector = mat_combos[:]
        if not self.symmetric:
            for combo in norm_combos:
                string_vector.append(combo)

        outputs = self.parallelize(string_vector)

        for i in range(len(mat_combos)):
            c = mat_coords[i]
            self.mat[c[0], c[1]] = outputs[i]

        if self.symmetric:
            self.mat = self.symmetrize(self.mat)
        else:
            self.test_normalization = outputs[len(mat_combos):]

    def parallelize(self, string_vector):
        pool = Pool(cpu_count())
        outputs = pool.map(self.redirect_to_kernel, string_vector)
        pool.close()
        pool.join()
        return outputs

    def generate_string_combos(self):
        """generate all string combinations required to build gram matrix
        as well as all norm combos"""
        mat_combos = []
        mat_coords = []
        norm_combos = []

        for row, s in enumerate(self.S):
            for col, t in enumerate(self.T):
                if self.symmetric and row > col:
                    pass
                else:
                    mat_combos.append([s, t])
                    mat_coords.append([row, col])

        # sort according to longest string
        zipped = zip(mat_combos, mat_coords)
        zipped_sorted = sorted(zipped, key=lambda x: len(x[0][0]) + len(x[0][1]), reverse=True)
        separated = list(zip(*zipped_sorted))
        mat_combos = list(separated[0])
        mat_coords = list(separated[1])

        if not self.symmetric:
            # need to calculate normalization values seperately
            for s in self.S:
                norm_combos.append([s,s])
        return mat_combos, mat_coords, norm_combos

    def redirect_to_kernel(self, sc):
        ret = kernel(sc[0], sc[1], self.n)
        return ret

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
        else:
            for idx, s in enumerate(self.S):
                self.test_normalization[idx] = self.kernel(s, s, self.n)

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
            # if not self.test_normalization[row]:
            #     self.test_normalization[row] = kernel(s, s, self.n)
            return self.mat[row, col] / sqrt(self.test_normalization[row] * self.train_normalization[col])

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

    # train_texts = ["'us export inspections thous bushels soybeans wheat corn'", 'us exporters report tonnes corn sold ussr']

    train_texts =  ['grain reserve holdings breakdown us agriculture department gave following breakdown grain remaining farmerowned grain reserve april mln bushels reserve number ii iii iv v vi wheat nil nil corn sorghumx barley x mln cwts note usda says totals may match total reserve numbers reuter',
                   # 'us exporters report tonnes corn sold ussr',
                   #'brazil coffee exports disrupted strike dayold strike brazilian seamen affecting coffee shipments could lead short term supply squeeze abroad exporters said could quantify much coffee delayed said least pct coffee exports carried brazilian ships movement foreign vessels also disrupted port congestion caused strike series labor disputes bad weather meant brazils coffee exports running average two weeks behind schedule since start year one source added end february shipments fallen bags behind registrations leaving around mln bags shipped march march bags shipped sources said given brazils port loading capacity around bags day even normal operations resumed immediately interrupted bad weather march registered coffee inevitably shipped april added reuter']
                   # 'us grain analysts see lower corn soy planting grain analysts surveyed american soybean association asa projected acreage year mln acres soybeans mln acres corn farmers planted mln acres soybeans mln acres corn according february usda supplydemand report usda release planting intentions report march survey included soybean estimates corn estimates released march soybean update newsletter sent members estimates ranged mln mln acres soybeans mln mln acres corn asa spokesman said association plans survey farmers planting intentions year reuter',
                   # 'union shippers agree cut ny port costs new york shipping association international longshoremens association said agreed cut cargo assessments port new york new jersey pct labor intensive cargos charges cargo handled union workers reduced dlrs ton dlrs ton effective april one according agreement union shippers assessments used fund workers benefits lowering price get bulk cargo flowing spokesman new york shipping association said',
                   'grain certificate redemptions put mln bu mln bushels government grain allocated redemptions commodity certificates since program began april according commodity credit corporation redemptions included mln bushels corn valued mln dlrs average perbushel price dlrs since current grain catalogs issued december ccc wheat redemptions totaled mln bushels valued mln dlrs since december']
                   # 'us exporters report tonnes corn switched unknown ussr']
                   # 'midwest cash grain slow country movement cash grain dealers reported slow country movement corn soybeans across midwest even corn sales pikandroll activity seen earlier week drying dealers said usda may adjust posted county price gulf take account high barge freight rates way keep corn sales flowing added current plan probably given weeks see work hoped corn soybean basis values continued drop illinois midmississippi river due strong barge freight rates toledo chicago elevators finishing loading first corn boats new shipping season supporting spot basis values terminal points corn soybeans toledo und may unc und may unc cincinnati und may unc ovr may new und may unc und may dn ne indiana und may unc ovr may dn chicago ovr may unc und may unc seneca und may dn und may unc davenport und may dn und may dn clinton und may dn ua cedar rapids und may dn und may dn hrw wheat toledo lb ovr may chicago lb ovr may unc cincinnati dp ovr may unc ne indiana dp ovr may unc pik certificates pct unc dn nc comparison ua unavailable unc unchanged dp delayed pricing reuter',
                   # 'brazil seamen continue strike despite court hundreds marines alert key brazilian ports seamen decided remain indefinite strike even higher labour court saturday ruled illegal union leaders said halt first national strike seamen years started february union leaders said would return work unless got pct pay rise shipowners offered per cent raise seamen rejected nothing lose want lay workers fine determined carry protest end union leader said said decided meeting marines take ships seamen would abandon vessels let marines handle situation spokesman rio de janeiro port said order send marines take ports given navy minister henrique saboya grounds ports areas national security said incidents strike cut exports imports made estimated ships idle petrol station owners four states also continued shutdown fears combination two stoppages could lead serious fuel shortage reuter']

    # build Gram matrix
    GC_train = GramCalc(train_texts, train_texts, n, kernel=kernel, symmetric=True)
    Gram_train_matrix = GC_train.calculate(parallel=False)
    print("in main")
    print("Gram train matrix", Gram_train_matrix)


    test_texts = ['grain certificate redemptions put mln']

    GC_test = GramCalc(test_texts, train_texts, n, kernel=kernel, symmetric=False)
    Gram_test_matrix = GC_test.calculate(parallel=False)
    print("Gram test matrix", Gram_test_matrix)

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
