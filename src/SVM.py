import numpy as np
from sklearn import svm
from math import sqrt

lam = 0.5

def K(n, sx, t):
    """Kernel that gives the sum over all common sub sequences weighted according to their frequency and length
    sx, t = string 1, 2
    n = max length of sub strings"""

    # stop recursion condition
    if min(len(sx), len(t)) < n:
        return 0

    # x is the last character in string sx
    x = sx[-1]
    # s is the string s without x
    s = sx[0: -1]

    # performs summation of K(n-1) prime
    sum = 0
    for j, tj in enumerate(t):
        if tj == x:
            sum += K_prime(n-1, s, t[0: j]) * (lam ** 2)

    return K(n, s, t) + sum


def K_prime(i, sx, t):
    """aiding function that:
    counts the length from the beginning of the the particular sequence through the end of string s and t
    instead of just counting the lengths"""
    if i == 0:
        return 1
    if min(len(sx), len(t)) < i:
        return 0

    s = sx[0: -1]
    return lam * K_prime(i, s, t) + K_double_prime(i, sx, t)


def K_double_prime(i, sx, t):
    """function to speed up calculation of K_prime_i"""
    if min(len(sx), len(t)) < i:
        return 0

    x = sx[-1]
    s = sx[0: -1]

    # if the two string share the same last letter, faster calculation
    if x == t[-1]:
        t = t[0: -1]
        return lam * (K_double_prime(i, sx, t) + lam * K_prime(i-1, s, t))

    # else perform the original K'' calculation
    else:
        sum = 0
        for j, tj in enumerate(t):
            if tj == x:
                sum += K_prime(i-1, s, t[0: j]) * lam ** (len(t) - j + 1)

        return sum


def K_norm(n, s, t):
    normalize = K(n, s, s) * K(n, t, t)
    return K(n, s, t) / sqrt(normalize)

def build_matrix(n, X, Y):
    Gram = np.zeros((len(X), len(Y)))
    for idx, s in enumerate(X):
        for jdx, t in enumerate(Y):
            Gram[idx, jdx] = K_norm(n, s, t)

    return Gram

def main():
    # original data
    X = ["science is organized knowledge", "one of these days im gonna get dead","one of these days im gonna get organisized", "wisdom is organized wisdom"]

    # labels
    labels = ["Kant", "Schopenhauer", "bob", "fred"]
    # labels in numerical form
    Y = np.arange(len(labels))

    # data to predict
    Z = ["one of these days im goa get dead","science is knowledge","one of these days im gonna get onisized", "wisdom is orgazed wisdom"]


    n = 2
    Gram = build_matrix(n, X, X)

    print("original gram")
    print(Gram)

    clf = svm.SVC(kernel='precomputed', probability=True)

    clf.fit(Gram, Y)

    new_Gram = build_matrix(n, Z, X)

    print("\nnew gram")
    print(new_Gram)

    print("\nprediction")
    print(clf.predict(new_Gram))

    print("""\narray-like, shape (n_samples, n_classes

    Returns the probability of the sample for each class in the model. The columns correspond to the classes in sorted order, as they appear in the attribute classes_.
""")
    print(clf.predict_proba(new_Gram))

    print("\nclasses")
    print(clf.classes_)

main()

