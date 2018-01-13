# SSK kernel - similarity measure between two strings based on non-contiguous substrings
# uses DP algorithm to speed things up

import numpy as np
from math import sqrt

lam = 0.5  # decay factor - penalizes non-contiguous substrings, value between 0 and 1


# DP method to find inner product between s and t
def K_memo(n, S, T):
    # K prime matrix
    Kp = np.zeros((n, len(S)+1, len(T)+1))

    # K double prime matrix
    Kpp = np.zeros((n, len(S)+1, len(T)+1))

    # 1st row
    for s in range(len(S)+1):
        for t in range(len(T)+1):
            Kp[0][s][t] = 1

    # iterate over all length of substrings
    for i in range(1, n):

        # iterate over every letter in s
        for s in range(1, len(S)+1):

            # iterate over every letter in t
            for t in range(1, len(T)+1):

                # if length of current substring is less than i
                if min(s, t) < i:

                    Kpp[i][s][t] = 0
                    Kp[i][s][t] = 0

                else:
                    # if last letter in T equals last letter in S
                    if S[s-1] == T[t-1]:
                        Kpp[i][s][t] = lam * (Kpp[i][s][t-1] + lam * Kp[i-1][s-1][t-1])

                    else:
                        # walk through string up to t
                        # j = all indices in t where the letter is x (the last letter in s)
                        sum = 0
                        for j, tj in enumerate(T[:t]):
                            if tj == S[s-1]:
                                sum += Kp[i-1][s-1][j] * lam ** (t - j + 1)

                        Kpp[i][s][t] = sum

                    Kp[i][s][t] = lam * Kp[i][s-1][t] + Kpp[i][s][t]

    # Final step
    K = np.zeros((len(S)+1, len(T)+1))

    # build the final kernel backwards instead of recursively
    for s in range(1, len(S)+1):

        for t in range(1, len(T)+1):

            if min(s, t) < n:
                K[s][t] = 0

            else:
                sum = 0
                for j, tj in enumerate(T[:t]):
                    if tj == S[s-1]:
                        sum += Kp[n-1][s-1][j] * lam ** 2

                K[s][t] = K[s-1][t] + sum

    # return the final kernel from the full strings
    return K[-1][-1]


def normkernel(n, S, T):
    print("\nstrings: ")
    print("s = ", S)
    print("t = ", T)
    print("\nfirst kernel executing...")
    k1 = K_memo(S, S, n)
    print("kernel(s, s, n) ", k1)
    print("done")
    print("\nsecond kernel executing...")
    k2 = K_memo(T, T, n)
    print("kernel(t, t, n) ", k2)
    print("done")
    print("\nlast kernel executing...")
    res = K_memo(S, T, n) / sqrt(k1 * k2)
    print("done. returning: ", res)
    return res


def main():
    kss = [0.580, 0.580, 0.478, 0.439, 0.406, 0.370]
    for x in range(1, 7):
        print(x, normkernel(x, "science is organized knowledge", "wisdom is organized life"), 'should be', kss[x - 1])

# main()
