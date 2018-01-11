# SSK kernel - similarity measure between two strings based on non-contiguous substrings
# uses DP algorithm to speed things up

import numpy as np
from math import sqrt


# DP method to find inner product between s and t
def kernel(s, t, n):

	# kprime[i, x, y] = k'_i(s[0:x+1], t[0:y+1])
	# kdoubleprime[i, x, y] = k''_i(s[0:x+1], t[0:y+1])
	kdoubleprimes = np.zeros((n, len(s), len(t)))
	kprimes = np.zeros((n, len(s), len(t)))

	# 1st row
	for s_ in range(len(s)):
		for t_ in range(len(t)):
			kprimes[0][s_][t_] = 1;

	for n_ in range(1, n):
		for s_ in range(len(s)):
			for t_ in range(len(t)):
				len_s, len_t = s_+1, t_+1
				if(min(len_s, len_t) < n_):
					kdoubleprimes[n_][s_][t_] = 0;
					kprimes[n_][s_][t_] = 0;
				else:
					x = s[s_]
					z = t[t_]
					if(x == z):
						kdoubleprimes[n_][s_][t_] = lambdaval * kdoubleprimes[n_][s_][t_-1] + lambdaval**2 * kprimes[n_-1][s_-1][t_-1]
					else:
						j = [pos for pos, char in enumerate(t[0:t_+1]) if char == x]
						kdoubleprimes[n_][s_][t_] = sum([kprimes[n_-1][s_-1][t_- j_] * lambdaval**(len_t-j_+1) for j_ in j]);

					kprimes[n_][s_][t_] = lambdaval * kprimes[n_][s_-1][t_] + kdoubleprimes[n_][s_][t_];
				
	# ks[x, y] = k_n(s[0:x+1], t[0:y+1])
	ks = np.zeros((len(s), len(t))) # K_n(s,t)
	for s_ in range(len(s)):
		for t_ in range(len(t)):
			len_s, len_t = s_+1, t_+1
			if(min(len_s, len_t) < n):
				ks[s_][t_] = 0;
			else:
				x = s[s_] # last letter
				j = [pos for pos, char in enumerate(t[0:s_+1]) if char == x]
				ks[s_][t_] = ks[s_-1][t_] + sum([kprimes[n-1][s_-1][j_-1] * lambdaval**2 for j_ in j]);

	# ks[len(s), len(t)] = k_n(s, t)
	return ks[-1][-1];

def normkernel(s, t, n):
	k1 = kernel(s,s,n)
	k2 = kernel(t,t,n)
	res = kernel(s,t,n) / sqrt(k1*k2)
	return res;



lambdaval = 0.5; # decay factor - penalizes non-contiguous substrings, value between 0 and 1

s = "cat"
t = "car"
n = 2


#### examples to check that its working ########
#print('k(car, car, 1) = ', kernel('car', 'car', 1), 'should be 3*lambda^2 = .75')
#print('k(car, car, 2) = ', kernel('car', 'car', 2), ' should be lambda^6 + 2*lambda^4 = 0.140625')
#print('k(car, car, 3) = ', kernel('car', 'car', 3), 'should be lambda^6 = 0.0156')

#print('normkernel(cat, car, 1) = ', normkernel('cat', 'car', 1), 'should be 2/3')
#print('kernel(cat, car, 2) = ', kernel('cat', 'car', 2), 'should be lambda^4 = 0.0625')
#print('normkernel(cat, car, 2) = ', normkernel('cat', 'car', 2), 'should be 1/(2+lambda^2) = 0.44444')

#print(kernel("AxxxxxxxxxB","AyB", 2), 'should be =0.5^14 = 0,00006103515625')

print(kernel("AxxxxxxxxxB","AxxxxxxxxxB", 2), 'should be 12.761724710464478')
print(kernel("ab","abb", 2), 'should be 0.5^5 + 0.5^4 = 0,09375')
print(kernel("ab","axb", 2), 'should be =0.5^5 = 0,03125')

#print(normkernel("ab","ab", 2), 'should be 1')
#print(normkernel("AxxxxxxxxxB","AxxxxxxxxxB", 2), 'should be 1')

# these dont work for some reason.
#kss = [0.580, 0.580, 0.478, 0.439, 0.406, 0.370]
#for x in range(1,7):
#	print(x, normkernel("science is organized knowledge", "wisdom is organized life", x), 'should be', kss[x-1])

###########################################


