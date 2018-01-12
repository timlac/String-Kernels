
#	String Subsequence kernel - similarity measure between two strings based on contiguous and non-contiguous substrings
#	uses dynamic programming algorithm for better performance

import numpy as np
from math import sqrt

lambdaval = 0.5  # decay factor - penalizes non-contiguous substrings, value between 0 and 1


""" Kernel that gives the sum over all common subsequences 
weighted according to their frequency and length
s, t = strings to be compared
n = length of substrings """
def kernel(s, t, n):

	#	calculate kprime and kdoubleprime for 
	#	all partial versions of s and t	
	kdoubleprimes = np.zeros((n, len(s), len(t)))
	kprimes = np.zeros((n, len(s), len(t)))

	#	1st row - only kprime is calculated
	for s_ in range(len(s)):
		for t_ in range(len(t)):
			kprimes[0][s_][t_] = 1;
			
	#	rest of the rows - both kprime and kdoubleprime 
	#	are calculated
	for i in range(1, n):
		for s_ in range(len(s)):
			for t_ in range(len(t)):
				len_s, len_t = s_+1, t_+1
				if(min(len_s, len_t) < i):
					kdoubleprimes[i][s_][t_] = 0;
					kprimes[i][s_][t_] = 0;
				else:
					x = s[s_]
					z = t[t_]
					if(x == z):
						kdoubleprimes[i][s_][t_] = lambdaval * kdoubleprimes[i][s_][t_-1] + lambdaval**2 * kprimes[i-1][s_-1][t_-1]
					else:
						j = [pos for pos, char in enumerate(t[0:t_+1]) if char == x]
						kdoubleprimes[i][s_][t_] = sum([kprimes[i-1][s_-1][j_-1] * lambdaval**(len_t-j_+1) for j_ in j]);

					kprimes[i][s_][t_] = lambdaval * kprimes[i][s_-1][t_] + kdoubleprimes[i][s_][t_];
			

	#	final step - calculate k_n(s, t)
	ks = np.zeros((len(s), len(t)))
	for s_ in range(len(s)):
		for t_ in range(len(t)):
			len_s, len_t = s_+1, t_+1
			if(min(len_s, len_t) < n):
				ks[s_][t_] = 0;
			else:
				x = s[s_] # last letter
				j = [pos for pos, char in enumerate(t[0:t_+1]) if char == x]
				ks[s_][t_] = ks[s_-1][t_] + sum([kprimes[n-1][s_-1][j_-1] * lambdaval**2 for j_ in j]);

	
	#	return k_n(s,t)
	return ks[-1][-1];
	
""" Normalized version of the kernel
s, t = strings to be compared
n = max length of sub strings """
def normkernel(s, t, n):
	k1 = kernel(s,s,n)
	k2 = kernel(t,t,n)
	res = kernel(s,t,n) / sqrt(k1*k2)
	return res;

""" Examples to check that it's working """
def test():

	print('k(car, car, 1) = ', kernel('car', 'car', 1), 'should be 3*lambda^2 = .75')
	print('k(car, car, 2) = ', kernel('car', 'car', 2), ' should be lambda^6 + 2*lambda^4 = 0.140625')
	print('k(car, car, 3) = ', kernel('car', 'car', 3), 'should be lambda^6 = 0.0156')

	print('normkernel(cat, car, 1) = ', normkernel('cat', 'car', 1), 'should be 2/3')
	print('kernel(cat, car, 2) = ', kernel('cat', 'car', 2), 'should be lambda^4 = 0.0625')
	print('normkernel(cat, car, 2) = ', normkernel('cat', 'car', 2), 'should be 1/(2+lambda^2) = 0.44444')

	print(kernel("AxxxxxxxxxB","AyB", 2), 'should be =0.5^14 = 0.00006103515625')
	print(kernel("AxxxxxxxxxB","AxxxxxxxxxB", 2), 'should be 12.761724710464478')

	print(kernel("ab","axb", 2), 'should be =0.5^5 = 0.03125')
	print(kernel("ab","abb", 2), 'should be 0.5^5 + 0.5^4 = 0.09375')
	print(normkernel("ab","ab", 2), 'should be 1')
	print(normkernel("AxxxxxxxxxB","AxxxxxxxxxB", 2), 'should be 1')

	kss = [0.580, 0.580, 0.478, 0.439, 0.406, 0.370]
	for x in range(1,7):
		print(x, normkernel("science is organized knowledge", "wisdom is organized life", x), 'should be', kss[x-1])



test()


