import numpy as np
import json
import os
from FLHeteroBackend import settings


def load_fed_weight():
    pass


def load_history(filename):
    with open(filename) as f:
        data = json.load(f)
    return data


def chebyshev_distance(x, y):
    return np.max(np.abs(x - y))


def euclidean_distance(x, y):
    return np.linalg.norm(x - y)


def feature_normalize(data):
    mu = np.mean(data)
    std = np.std(data)
    return (data - mu) / std


def feature_standardize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def grid(x, n=10, return_quantiles=False):
    step = 100 // n
    percentiles = np.arange(0, 100, step) + step
    quantiles = [np.percentile(x, q, interpolation='lower') for q in percentiles]
    quantiles = np.unique(quantiles)
    y = [np.searchsorted(quantiles, val, 'left') for val in x]
    if return_quantiles:
        return y, quantiles
    return y
