from typing import List

def prod_non_zero_diag(X: List[List[int]]) -> int:
    S = 1
    f = False
    for i in range(min(len(X), len(X[0]))):
        if X[i][i]:
            S *= X[i][i]
            f = True
    if f:
        return S
    else:
        return -1

def are_multisets_equal(x: List[int], y: List[int]) -> bool:
    if len(x) != len(y):
        return False
    x = list(x)
    y = list(y)
    x.sort()
    y.sort()
    for i in range(len(x)):
        if x[i] != y[i]:
            return False
    return True

def max_after_zero(x: List[int]) -> int:
    max_x = min(x)
    f = False
    for i in range(len(x) - 1):
        if (x[i] == 0) and (x[i+1] > max_x):
            max_x = x[i+1]
            f = True
    if f:
        return max_x
    else:
        return -1

def convert_image(image: List[List[List[float]]], weights: List[float]) -> List[List[float]]:
    X = []
    for i in image:
        Y = []
        for j in i:
            S = 0
            for k in range(len(j)):
                S += j[k]*weights[k]
            Y.append(S)
        X.append(Y)
    return X

def run_length_encoding(x: List[int]) -> (List[int], List[int]):
    X = []
    Y = []
    sym = x[0]
    num = 0
    for i in range(len(x)):
        if x[i] == sym:
            num += 1
        else:
            X.append(sym)
            Y.append(num)
            sym = x[i]
            num = 1
    X.append(sym)
    Y.append(num)
    return (X, Y)

def pairwise_distance(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    res = []
    for i in X:
        Z = []
        for j in Y:
             S = 0
             for k in range(len(j)):
                 S += (i[k] - j[k]) ** 2
             Z.append(S ** 0.5)
        res.append(Z)
    return res