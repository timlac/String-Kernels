####
####		Introducing the approximated version of the kernel


import numpy as np
import itertools
from math import sqrt

from sklearn.feature_extraction.text import CountVectorizer


from SSK import normkernel, kernel
from process_file import process_directory, process_file_without_modeapte
from preprocessing import process_file




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
# TODO: test this function
def ngram_extraction(X, n):
	pattern = '(?u)[a-z ]*' # only accept a-z and space
	
	# TODO: introduce some way so that only n grams including [a-z] and whitespace are included. my regex patterns seem to be overriden since they dont make any difference.
	# could tokenize away numbers
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
		one = kernel(x, s_i, n)
		two = kernel(z, s_i, n)
		res +=  one * two;
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
	
	kerneltypes = [None, 'approx']
	
	S = form_S(X, n, k)	#	form set of k most common n-grams in the data set
	
	print(S, len(S), 'the set')
	print('building gram matrix ...')
	GRAM1 = create_gram_matrix(n, S, X, X, kerneltype=kerneltypes[0])
	print('building gram matrix ...')
	print(GRAM1, 'normal SSK')
	GRAM2 = create_gram_matrix(n, S, X, X, kerneltype=kerneltypes[1])
	print(GRAM2, 'approximated SSK')
	similarity = gram_matrix_alignment(n, S, X, X, kerneltypes); #	measure of similarity between gram matrix made by SSK and by approximated version of the kernel
	
	print('similarity between gram matrices = ', similarity)


## to be continued
def experiment():

	train_index, texts = process_directory()
	wholedataset = getData(train_index, texts)
	term_doc_matrix, names = ngram_extraction(wholedataset, 3)
	print(term_doc_matrix.shape, 'should be (21578, 8727) according to the report, however, they might have different stop words/different preprocessing and they dont seem to include numbers in their n-grams')

	categories = ['earn', 'acq', 'money-fx', 'grain', 'crude', 'trade', 'interest', 'ship', 'wheat', 'corn']
	
	print("## approximated SSK")
	


#test()
experiment()







