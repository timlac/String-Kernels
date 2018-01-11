# code for the kernel - similarity measure between two strings based on non-contiguous substrings
# obs: this document i made mostly to understand the basics of the kernel, its a "naive" implementation

import numpy as np
from math import sqrt


# find all non-contiguous substrings of length k,
# and their indices
def get_all_substrings(string, k):
    length = len(string)
    substr_list = []

    for i in range(length):
        for j in range(i + 1, length):
            substr = string[i] + string[j:j + k - 1]
            if len(substr) == k and substr not in substr_list:
                substr_list.append(substr)
    return substr_list


def find_indices(string, substr):
    length = len(string)
    len_ = len(substr)
    i_list = []

    for i in range(length):
        for j in range(i, length):
            j = j + 1
            substring = string[i] + string[j:j + len_ - 1]
            index2 = [j_ for j_ in range(j, min(length, j + len_ - 1))]
            index = [i] + index2

            if substring == substr and index not in i_list:
                i_list.append(index)

    return i_list


# s = string 1, t = string 2
# K_i(s, t) = inner product between s & t
# i = length of substring
def k(s, t, i):
    if i == 0:
        return 1

    if min(len(s), len(t)) < i:
        return 0

    substr_s = get_all_substrings(s, i)
    substr_t = get_all_substrings(t, i)

    all_substr = substr_s
    for sub in substr_t:
        if sub not in all_substr:
            all_substr.append(sub)

    res = 0
    for substr in all_substr:
        index_s = find_indices(s, substr)
        index_t = find_indices(t, substr)
        for i_s in index_s:
            len1 = i_s[-1] - i_s[0] + 1
            for i_t in index_t:
                len2 = i_t[-1] - i_t[0] + 1

                res += lambdaval ** (len1 + len2)
    return res


# normalized version of K_i(s, t)
def normk(s, t, i):
    if i == 0:
        return 1

    if min(len(s), len(t)) < i:
        return 0

    k1 = k(s, s, i)
    k2 = k(t, t, i)
    res = k(s, t, i) / sqrt(k1 * k2)

    return res


lambdaval = .5  # decay factor - penalizes non-contiguous substrings, value between 0 and 1

# print(get_all_substrings("science is organized knowledge", 1), '*')
# print(find_indices("catca", "ca"))
# print(find_indices("cat", "car"))
# print(find_indices("science is organized knowledge", "s"))

# print(find_indices("wisdom is organized life", "s"))

# print(1, normk("go knowledge", "go life", 2))


# print(k("cat","cat",2))
# print('k(cat, car, 2) = ', normk("cat","car", 2), 'should be 0.44444444 = 1/(2+lambda^2) (normalized)')
# print('k(fog, fob, 2) = ', k("fog","fob", 2))
##print('k(bat, car, 2) = ', normk("bat","car", 2), 'should be 0 (normalized)')

## the ones below don't work and im not sure why (ie they dont have the same results as in the report)

# print(1, k("science is organized knowledge", "science is organized knowledge", 1))
# print(1, k("wisdom is organized life", "wisdom is organized life", 1))

# print(1, normk("science is organized knowledge", "wisdom is organized life", 2))

ks = [0.580, 0.580, 0.478, 0.439, 0.406, 0.370]

s = "science is organized knowledge"
t = "wisdom is organized life"

for i in range(2, 7):
    K = normk(s, t, i)
    print(i, K, 'should be', ks[i - 1])

#	print(x, normk("science knowledge", "wisdom life", x), 'should be', ks[x-1])
