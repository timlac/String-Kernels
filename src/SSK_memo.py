# SSK kernel - similarity measure between two strings based on non-contiguous substrings
# uses DP algorithm to speed things up

import numpy as np
from math import sqrt

lambdaval = 0.5  # decay factor - penalizes non-contiguous substrings, value between 0 and 1

# print('k(cat, car, 2) = ', k('cat', 'car', 2))

s = "cat"
t = "car"
n = 2

#	compute k(cat, car, 2) with memoization - ¡might be more efficient ways to do this but it was the first i thought of!

kdoubleprimes = np.zeros((n, len(s) + 1, len(t) + 1))
kprimes = np.zeros((n, len(s) + 1, len(t) + 1))


# 1st row
for s_ in range(len(s) + 1):
    for t_ in range(len(t) + 1):
        kprimes[0][s_][t_] = 1

for n_ in range(1, n):
    for s_ in range(len(s) + 1):
        for t_ in range(len(t) + 1):
            print(s_, t_, s[0:s_ - 1], t[0:t_ - 1], '*')
            if (min(s_, t_) < n_):
                kdoubleprimes[n_][s_][t_] = 0
                kprimes[n_][s_][t_] = 0
            else:
                kdoubleprimes[n_][s_][t_] = lambdaval * kdoubleprimes[n_][s_][t_ - 1] + lambdaval ** 2 * \
                                            kprimes[n_ - 1][s_ - 1][t_ - 1]
                kprimes[n_][s_][t_] = lambdaval * kprimes[n_][s_ - 1][t_] + kdoubleprimes[n_][s_][t_]

ks = np.zeros((len(s) + 1, len(t) + 1))  # K_2(s,t)
for s_ in range(len(s) + 1):
    for t_ in range(len(t) + 1):
        if (min(s_, t_) < n_):
            ks[s_][t_] = 0
        else:
            x = s[min(len(s) - 1, s_ - 1)]  # last letter
            j = [pos for pos, char in enumerate(s) if char == x]
            ks[s_][t_] = ks[s_ - 1][t_] + lambdaval ** 2 * sum([kprimes[n - 1][s_ - 1][t_ - j_] for j_ in j])


print(kdoubleprimes, 'kdoubleprimes')
print(kprimes, 'kprimes')
print(ks, 'ks')

print('k(ca, car, 2) = ', ks[-1][-1], 'unnormalized')
