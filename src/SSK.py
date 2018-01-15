#	String Subsequence kernel - similarity measure between two strings based on contiguous and non-contiguous substrings
#	uses dynamic programming algorithm for better performance

import numpy as np
from math import sqrt

lam = 0.5  # decay factor - penalizes non-contiguous substrings, value between 0 and 1
""" Kernel that gives the sum over all common subsequences 
weighted according to their frequency and length
s, t = strings to be compared
n = length of substrings """


def kernel(S, T, n):
    if min(len(S), len(T)) < n:
        return 0

    # K prime matrix
    Kp = np.zeros((n, len(S) + 1, len(T) + 1))

    # K double prime matrix
    Kpp = np.zeros((n, len(S) + 1, len(T) + 1))

    # 1st row
#    for s in range(len(S) + 1):
#        for t in range(len(T) + 1):
#            Kp[0][s][t] = 1
    Kp[0,:,:] = 1

    # iterate over all length of substrings
    for i in range(1, n):

        # iterate over every letter in s
        for s in range(1, len(S) + 1):

            # iterate over every letter in t
            for t in range(1, len(T) + 1):

                # if length of current substring is less than i
#                if min(s, t) < i:

#                    Kpp[i][s][t] = 0
#                    Kp[i][s][t] = 0

#                else:
                if min(s, t) >= i:
                    # if last letter in T equals last letter in S
                    if S[s - 1] == T[t - 1]:
                        Kpp[i][s][t] = lam * (
                            Kpp[i][s][t - 1] + lam * Kp[i - 1][s - 1][t - 1])

                    else:
                        # walk through string up to t
                        # j = all indices in t where the letter is x (the last letter in s)
                        sum = 0
                        for j, tj in enumerate(T[:t]):
                            if tj == S[s - 1]:
                                sum += Kp[i - 1][s - 1][j] * lam**(t - j + 1)

                        Kpp[i][s][t] = sum

                    Kp[i][s][t] = lam * Kp[i][s - 1][t] + Kpp[i][s][t]

    # Final step
    K = np.zeros((len(S) + 1, len(T) + 1))

    # build the final kernel backwards instead of recursively
    for s in range(1, len(S) + 1):

        for t in range(1, len(T) + 1):
#
#            if min(s, t) < n:
#                K[s][t] = 0
#
#            else:
            if min(s, t) >= n:
                sum = 0
                for j, tj in enumerate(T[:t]):
                    if tj == S[s - 1]:
                        sum += Kp[n - 1][s - 1][j] * lam**2

                K[s][t] = K[s - 1][t] + sum

    # return the final kernel from the full strings
    return K[-1][-1]


""" Normalized version of the kernel
s, t = strings to be compared
n = max length of sub strings """


def normkernel(S, T, n):
    print("\nstrings: ")
    print("s = ", S)
    print("t = ", T)
    print("\nfirst kernel executing...")
    k1 = kernel(S, S, n)
    print("kernel(s, s, n) ", k1)
    print("done")
    print("\nsecond kernel executing...")
    k2 = kernel(T, T, n)
    print("kernel(t, t, n) ", k2)
    print("done")
    print("\nlast kernel executing...")
    res = kernel(S, T, n) / sqrt(k1 * k2)
    print("done. returning: ", res)
    return res


""" Examples to check that it's working """


def test():

    print('k(car, car, 1) = ', kernel('car', 'car', 1),
          'should be 3*lambda^2 = .75')
    print('k(car, car, 2) = ', kernel('car', 'car', 2),
          ' should be lambda^6 + 2*lambda^4 = 0.140625')
    print('k(car, car, 3) = ', kernel('car', 'car', 3),
          'should be lambda^6 = 0.0156')

    print('normkernel(cat, car, 1) = ', normkernel('cat', 'car', 1),
          'should be 2/3')
    print('kernel(cat, car, 2) = ', kernel('cat', 'car', 2),
          'should be lambda^4 = 0.0625')
    print('normkernel(cat, car, 2) = ', normkernel('cat', 'car', 2),
          'should be 1/(2+lambda^2) = 0.44444')

    print(
        kernel("AxxxxxxxxxB", "AyB", 2),
        'should be =0.5^14 = 0.00006103515625')
    print(
        kernel("AxxxxxxxxxB", "AxxxxxxxxxB", 2),
        'should be 12.761724710464478')

    print(kernel("ab", "axb", 2), 'should be =0.5^5 = 0.03125')
    print(kernel("ab", "abb", 2), 'should be 0.5^5 + 0.5^4 = 0.09375')
    print(normkernel("ab", "ab", 2), 'should be 1')
    print(normkernel("AxxxxxxxxxB", "AxxxxxxxxxB", 2), 'should be 1')

    kss = [0.580, 0.580, 0.478, 0.439, 0.406, 0.370]
    for x in range(1, 7):
        print(x,
              normkernel("science is organized knowledge",
                         "wisdom is organized life", x), 'should be',
              kss[x - 1])


if __name__ == '__main__':
    test()
