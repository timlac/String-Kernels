# SSK kernel - uses recursion

import numpy as np
from math import sqrt

lambdaval = 0.5  # decay factor - penalizes non-contiguous substrings, value between 0 and 1

# K''_i(s,t)
def kdoubleprime(s, t, i):
    if min(len(s), len(t)) < i:
        return 0

#    print(s, t, i, '*')
    x = s[-1]
    if(len(t) > 0):
    	z = t[-1]
    else:
    	z = ""

    if x == z:
        res = lambdaval * (kdoubleprime(s, t[0:-1], i) + lambdaval * kprime(s[0:-1], t[0:-1], i - 1))
        return res
    else:
        # k__ = doubleprime(s, t[0:-1])
        k__ = 0
        j = []  # j:tj = x
        for m in range(len(t)):
            if t[m] == x:
                j.append(m)

        for j_ in j:
            k__ += kprime(s[0:-1], t[0:j_], i - 1) * lambdaval ** (len(t) - j_ + 1)

        return k__

# K'_i(s,t)
def kprime(s, t, i):

    if i <= 0:
        return 1

    if min(len(s), len(t)) < i:
        return 0

    exp1 = lambdaval * kprime(s[0:-1], t, i)
    exp2 = kdoubleprime(s, t, i)

    return (exp1 + exp2)


# K_i(s, t) = inner product between s & t
# i = length of substring
def k(s, t, i):
    if (min(len(s), len(t)) < i):
        return 0

    x = s[-1]

    exp1 = k(s[0:-1], t, i)
    exp2 = 0

    j = []  # j:tj = x
    for m in range(len(t)):
        if t[m] == x:
            j.append(m)
    for j_ in j:
        exp2 += kprime(s[0:-1], t[0:j_], i - 1) * lambdaval ** 2
    return (exp1 + exp2)


def normk(s, t, i):
    k1 = k(s, s, i)
    k2 = k(t, t, i)
    res = k(s, t, i) / sqrt(k1 * k2)
    return res




#### examples to check that its working ########
#print('k(car, car, 1) = ', k('car', 'car', 1), 'should be 3*lambda^2 = .75')
#print('k(car, car, 2) = ', k('car', 'car', 2), ' should be lambda^6 + 2*lambda^4 = 0.140625')
#print('k(car, car, 3) = ', k('car', 'car', 3), 'should be lambda^6 = 0.0156')

#print('normk(cat, car, 1) = ', normk('cat', 'car', 1), 'should be 2/3')
#print('k(cat, car, 2) = ', k('cat', 'car', 2), 'should be lambda^4 = 0.0625')
#print('normk(cat, car, 2) = ', normk('cat', 'car', 2), 'should be 1/(2+lambda^2) = 0.44444')

#print(k("AxxxxxxxxxB","AyB", 2), 'should be =0.5^14 = 0,00006103515625')

#print(k("ab","abb", 2), 'should be 0.5^5 + 0.5^4 = 0,09375')

#print(k("AxxxxxxxxxB","AxxxxxxxxxB", 2), 'should be 12.761724710464478')
#print(k("ab","axb", 2), 'should be =0.5^5 = 0,03125')

#print(normk("ab","ab", 2), 'should be 1')
#print(normk("AxxxxxxxxxB","AxxxxxxxxxB", 2), 'should be 1')

# these dont work for some reason.
kss = [0.580, 0.580, 0.478, 0.439, 0.406, 0.370]
for x in range(1,7):
	print(x, normk("science is organized knowledge", "wisdom is organized life", x), 'should be', kss[x-1])

###########################################

n = 2
s = "ab"
t = "abb"

kdoubleprimes = np.zeros((n, len(s), len(t)))
kprimes = np.zeros((n, len(s), len(t)))

for n_ in range(1, n):
	for s_ in range(len(s)):
		for t_ in range(len(t)):
#			print(s[0:s_+1], t[0:t_+1])
			kdoubleprimes[n_][s_][t_] = kdoubleprime(s[0:s_+1], t[0:t_+1], n_);
			kprimes[n_][s_][t_] = kprime(s[0:s_+1], t[0:t_+1], n_);


#print(kprimes, 'kprimes')
#print(kdoubleprimes, 'kdoubleprimes')


#print(kdoubleprime("c","c", 0))

ks = np.zeros((len(s), len(t))) # K_n(s,t)
for s_ in range(len(s)):
	for t_ in range(len(t)):
#		print(s[0:s_+1], t[0:t_+1], n)
		ks[s_][t_] = k(s[0:s_+1], t[0:t_+1], n)

#print(ks, 'ks')

