####
####		Approximated version of the kernel


import numpy as np
import itertools
from math import sqrt

from sklearn.feature_extraction.text import CountVectorizer

from SSK import normkernel, kernel
from preprocessing import process_directory, process_file, preprocess_regex



""" frobenius inner product between two gram matrices """
def frobenius(K1, K2):

	if(len(K1) != len(K2) or len(K1[0]) != len(K2[0])):
		print('invalid dimension for frobenius inner product')
		return -1;
	
	res = 0
	for i in range(len(K1)):
		for j in range(len(K1[0])):
			res += K1[i][j] * K2[i][j]
	return res;


"""	kernel matrix for the sample S using kernel ki """
def create_gram_matrix(n, S, X, Y, kerneltype=None):
	
	Gram = np.zeros((len(X), len(Y)))
	for idx, s in enumerate(X):
		for jdx, t in enumerate(Y):
			if(kerneltype is None):
				Gram[idx, jdx] = normkernel(s, t, n)
			elif(kerneltype=="approx"):
				Gram[idx, jdx] = normapproxkernel(s, t, n, S)				
			else:
				Gram[idx, jdx] = -1 # not supported
	return Gram

	

""" The (empirical) alignment of a kernel k1 with a kernel k2 with
respect to the sample S
n = length of substrings considered
S = sample of n-grams
X, Y = data sets
kerneltypes = [kerneltype for k1, kerneltype for k2],
where the possible kerneltypes are None = SSK and 'approx' = approximated kernel """
def gram_matrix_alignment(n, S, X, Y, kerneltypes):

	print('building gram matrix ...')
	K1 = create_gram_matrix(n, S, X, Y, kerneltypes[0])
	print('building gram matrix ...')
	K2 = create_gram_matrix(n, S, X, Y, kerneltypes[1])
	
	K12 = frobenius(K1, K2)
	K11 = frobenius(K1, K1)
	K22 = frobenius(K2, K2)
	
	res = K12 / sqrt(K11*K22)
	
	return res;


""" Extracts all n-grams from a data set """
def ngram_extraction(X, n):

	ngram_vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n)) 
	term_doc_matrix = ngram_vectorizer.fit_transform(X)
	names = ngram_vectorizer.get_feature_names()
	
	return term_doc_matrix, names;


""" We choose the k (contiguous) strings of length n which occur 
most frequently in the dataset and this forms our set S̃. """
def form_S(X, n, k):

	term_doc_matrix, ngrams = ngram_extraction(X, n)
	feature_vectors = term_doc_matrix.toarray()
	
	total = feature_vectors.sum(axis=0)
	sorted_tot = np.argsort(total)[::-1] # sorted in descending order
	most_common = [ngrams[sorted_tot[i]] for i in range(k)]
	
	return most_common;


""" Generates all possible 27^n n-grams.
Obs Not sure if this function is needed  """
def ngram_generation(n):
	alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", \
	"l", "m", "n", "o", "p", "q", "r", "t", "s", "u", "v", "w", "x", "y", "z", " "]	
	ngrams = itertools.product(alphabet, repeat = n)
	ngrams = list(ngrams)
	ngrams = [''.join(tup) for tup in ngrams]
	return ngrams;


""" approximated version of the SSK kernel
x and z documents (or strings)
n = length of substrings considered
S = set of n-grams """
def approxkernel(x, z, n, S):
	res = 0
	for s_i in S:
		xs = kernel(x, s_i, n)
		zs = kernel(z, s_i, n)
		res +=  xs * zs;
	return res;

""" normalized, approximated, version of the SSK kernel """
def normapproxkernel(x, z, n, S):
	xx = approxkernel(x,x,n,S)
	zz = approxkernel(z,z,n,S)
	xz = approxkernel(x,z,n,S)
	norm = xz / sqrt(xx * zz);
	return norm;

def getData(documents, texts):
	data = []
	for doc in documents:
		data.append(texts[doc])
	return data


# this one is a bit moot
def get_hundred_first_docs():
	# we are interested in all of the reuters data set here, not only docs in modeapte (as i understand it)
	train_index, texts = process_file_without_modeapte('../data/reut2-000.sgm')
	index = [i+1 for i in range(100)]
	X = getData(index, texts);
	
	return X;
	


def test():	
	
	n = 3
	k = 20
	
	# lil data set
	X = ["The support vector machine (SVM) is a powerful learning algorithm, e.g., for classification and clustering tasks, that works even for complex data structures such as strings, trees, lists and general graphs.", 
	"It is based on the usage of a kernel function for measuring scalar products between data units.", 
	"For analyzing string data Lodhi et al. (J Mach Learn Res 2:419–444, 2002) have introduced a String Subsequence kernel (SSK)"]
	
	X = [preprocess_regex(x) for x in X]
	
	term_doc_matrix, names = ngram_extraction(X, n)
	max_ngrams = len(names);

	kerneltypes = [None, 'approx']

	S = form_S(X, n, max_ngrams)	#	form set of k most common n-grams in the data set	
	for k in range(10, max_ngrams, 10):

		currentS = S[0:k];
		print(currentS, len(currentS), 'current set')
		similarity = gram_matrix_alignment(n, currentS, X, X, kerneltypes); #	measure of similarity between gram matrix made by SSK and by approximated version of the kernel
	
		print('k = ', k, 'similarity between gram matrices = ', similarity)


## to be continued
## preliminary experiment
def experiment():

	n = 5
	num_features = 1000
	categories = ['earn', 'acq', 'ship', 'corn']

	train, test, titles, texts, classes = process_directory()
	wholedataset = getData(train, texts)
	term_doc_matrix, names = ngram_extraction(wholedataset, n)
	S = form_S(wholedataset, n, num_features)	# form set of k most common n-grams in the data set
	gram = create_gram_matrix(n, S, wholedataset, wholedataset, kerneltype='approx')

	
		
	# not sure why its not the same here. might be a preprocessing thing, 
	# but they seem to mention that they only do removal of stopwords which wouldnt explain why i get MORE trigrams than them
	print('num unique trigrams = ', term_doc_matrix.shape[1], 'should be 8727 according to the report')

	print("## approximated SSK")
	


test()
#experiment()







