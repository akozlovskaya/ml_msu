import numpy as np
import sklearn
from sklearn.metrics import pairwise_distances


def silhouette_score(X, labels):
    u, indices = np.unique(labels, return_inverse=True)
    labels = np.arange(len(u))[indices]
    label_freqs = np.bincount(labels)
    pwd = pairwise_distances(X)

    clust_dists = np.zeros((len(pwd), len(label_freqs)), dtype=pwd.dtype)
    for i in range(len(pwd)):
        clust_dists[i] += np.bincount(labels, weights=pwd[i], minlength=len(label_freqs))

    intra_index = (np.arange(len(pwd)), labels)
    intra_clust_dists = clust_dists[intra_index]
    clust_dists[intra_index] = np.inf
    clust_dists /= label_freqs
    inter_clust_dists = clust_dists.min(axis=1)

    denom = (label_freqs - 1).take(labels, mode="clip")
    with np.errstate(divide="ignore", invalid="ignore"):
        intra_clust_dists /= denom

    sil_samples = inter_clust_dists - intra_clust_dists
    with np.errstate(divide="ignore", invalid="ignore"):
        sil_samples /= np.maximum(intra_clust_dists, inter_clust_dists)

    return np.mean(np.nan_to_num(sil_samples))


def get_unique(X):
    _, counts = np.unique(X, return_counts=True)
    unique = np.zeros(X.shape)
    unique[:counts.shape[0]] = counts
    return unique


def precision_recall(X, idx, denom, size):
    return np.sum(np.sum(X ** 2, axis=0)[idx] / denom) / size


def bcubed_score(true_labels, predicted_labels):

    lab_max = np.max([true_labels, predicted_labels]) + 1
    true_labels[true_labels == 0] = lab_max
    predicted_labels[predicted_labels == 0] = lab_max
    unique = np.unique(np.concatenate([true_labels, predicted_labels]))
    N = len(unique)
    M = len(true_labels)

    mas_true_lab = np.repeat([true_labels], N, axis=0)
    mas_pred_lab = np.repeat([predicted_labels], N, axis=0)

    mask_true = mas_true_lab == unique[:, np.newaxis]
    mask_pred = mas_pred_lab == unique[:, np.newaxis]
    nonzero_true = np.count_nonzero(mask_true, axis=1)
    nonzero_pred = np.count_nonzero(mask_pred, axis=1)
    idx_true = np.where(nonzero_pred != 0)[0]
    idx_pred = np.where(nonzero_true != 0)[0]
    nonzero_pred = nonzero_pred[idx_true]
    nonzero_true = nonzero_true[idx_pred]

    min_lab = np.min(unique)
    val = -1 if min_lab >= 0 else min_lab - 1
    mas_pred_lab[np.logical_not(mask_true)] = val
    mas_true_lab[np.logical_not(mask_pred)] = val

    mas_pred_lab = np.apply_along_axis(get_unique, 1, mas_pred_lab).T
    mas_true_lab = np.apply_along_axis(get_unique, 1, mas_true_lab).T

    if np.any(predicted_labels != predicted_labels[0]):
        mas_true_lab = mas_true_lab[1:]

    if np.any(true_labels != true_labels[0]):
        mas_pred_lab = mas_pred_lab[1:]

    precision = precision_recall(mas_true_lab, idx_true, nonzero_pred, M)
    recall = precision_recall(mas_pred_lab, idx_pred, nonzero_true, M)

    score = 2 * (precision * recall) / (precision + recall)
    return score


class KMeansClassifier(sklearn.base.BaseEstimator):
    def __init__(self, n_clusters):
        super().__init__()
        self.n_clusters = n_clusters

    def fit(self, data, labels):
        return self

    def predict(self, data):
        return None

    def _best_fit_classification(self, cluster_labels, true_labels):
        unique, index = np.unique(cluster_labels, return_index=True)
        clear_labels = true_labels[true_labels != -1]
        classes = []
        for lab in unique:
            clust_lab = true_labels[cluster_labels == lab]
            if len(clust_lab[clust_lab != -1]) == 0:
                un, counts = np.unique(clear_labels, return_counts=True)
            else:
                un, counts = np.unique(clust_lab[clust_lab != -1], return_counts=True)
            classes += [un[np.argmax(counts)]]

        mapping = np.zeros(self.n_clusters) - 1
        for i in cluster_labels[index]:
            mapping[i] = classes[i - np.min(cluster_labels[index])]

        predicts = mapping[cluster_labels]
        unique, counts = np.unique(clear_labels, return_counts=True)
        mapping[mapping == -1] = unique[np.argmax(counts)]

        return mapping, predicts
