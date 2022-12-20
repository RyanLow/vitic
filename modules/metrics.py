import numpy as np
import sklearn.metrics as metrics
from scipy.optimize import linear_sum_assignment


def NMI(labels_true, labels_pred):
    """Normalized Mutual Information metric"""
    return metrics.normalized_mutual_info_score(labels_true, labels_pred)


def ARI(labels_true, labels_pred):
    """Adjusted Rand Index metric"""
    return metrics.adjusted_rand_score(labels_true, labels_pred)


def FMI(labels_true, labels_pred):
    """Fowlkes Mallows Index metric"""
    return metrics.fowlkes_mallows_score(labels_true, labels_pred)


def ACC(labels_true, labels_pred):
    """Cluster Accuracy metric (using Hungarian method)"""
    D = max(labels_true.max(), labels_pred.max()) + 1
    n = len(labels_true)
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(n):
        w[labels_true[i], labels_pred[i]] += 1
    indices = linear_sum_assignment(w.max() - w)
    indices = np.asarray(indices).T
    return sum([w[i,j] for i,j in indices]) / n
