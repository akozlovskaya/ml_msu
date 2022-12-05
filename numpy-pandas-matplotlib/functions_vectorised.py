import numpy as np

def prod_non_zero_diag(X: np.ndarray) -> int:
    d = np.diag(X)
    nzd = np.array(d[d > 0])
    if len(nzd) > 0 :
        return nzd.prod()
    else:
        return -1


def are_multisets_equal(x: np.ndarray, y: np.ndarray) -> bool:
    return np.array_equal(np.sort(x), np.sort(y))


def max_after_zero(x: np.ndarray) -> int:
    mask = x[:-1] == 0
    y = x[1:][mask]
    if len(y) > 0:
        return y.max()
    else:
        return -1
        

def convert_image(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(image * weights, axis = 2)


def run_length_encoding(x: np.ndarray) -> (np.ndarray, np.ndarray):
    mask = np.roll(x, 1) != x
    mask[0] = True
    num = np.where(mask == True)[0]
    num_2 = num[1:]
    num_2 = np.append(num_2, len(x))
    return (x[mask], num_2 - num)


def pairwise_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.sqrt((X*X).sum(axis=1).reshape(-1, 1) + (Y*Y).sum(axis=1) - 2*X.dot(Y.T))
