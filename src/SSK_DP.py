# SSK kernel - uses recursion

import numpy as np
from math import sqrt

# K''_i(s,t)
def kdoubleprime(s, t, i):

	if(min(len(s), len(t)) < i):
		return 0;
	
	x = s[-1];
	z = t[-1];
	
	if(x == z):
		res = lambdaval*(kdoubleprime(s, t[0:-1], i) + lambdaval * kprime(s[0:-1], t[0:-1], i-1))
		return res;
	else:
		#k__ = doubleprime(s, t[0:-1])
		k__ = 0;
		j = [] #	j:tj = x
		for m in range(len(t)):
			if(t[m] == x):
				j.append(m);

		for j_ in j:			
			k__ += kprime(s, t[0:j_], i-1) * lambdaval**(len(t) - j_ + 1);

		return k__
	

# K'_i(s,t)
def kprime(s, t, i):
	if(i <= 0):
		return 1;

	if(min(len(s), len(t)) < i):
		return 0;

	exp1 = lambdaval * kprime(s[0:-1], t, i)
	exp2 = kdoubleprime(s, t, i)
	
	return (exp1 + exp2);

# K_i(s, t) = inner product between s & t
# i = length of substring
def k(s, t, i):
	if(min(len(s), len(t)) < i):
		return 0;
		
	x = s[-1];
		
	exp1 = k(s[0:-1], t, i) 
	exp2 = 0

	j = [] #	j:tj = x
	for m in range(len(t)):
		if(t[m] == x):
			j.append(m);
	for j_ in j:
		exp2 += kprime(s[0:-1], t[0:j_], i-1) * lambdaval**2;
	return (exp1 + exp2);

def normk(s, t, i):
	k1 = k(s,s,i)
	k2 = k(t,t,i)
	res = k(s,t, i) / sqrt(k1*k2)
	return res;
	

lambdaval = 0.5; # decay factor - penalizes non-contiguous substrings, value between 0 and 1



print('k(cat, car, 2) = ', k('cat', 'car', 2), 'should be lambda^4 = 0.0625')
print('k(car, car, 2) = ', k('car', 'car', 2), ' should be lambda^6 + 2*lambda^4 = 0.140625')
print('normk(cat, car, 2) = ', normk('cat', 'car', 2), 'should be 1/(2+lambda^2) = 0.44444')
#print('k(ca, car, 2) = ', k('ca', 'car', 2))
#print('k(cat, car, 1) = ', k('cat', 'car', 1))


kss = [0.580, 0.580, 0.478, 0.439, 0.406, 0.370]

#for x in range(1,7):
#	print(x, normk("science is organized knowledge", "wisdom is organized life", x), 'should be', kss[x-1])
	









