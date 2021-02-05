import json
import os

import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

from FLHeteroBackend import settings
from pca.cpca import CPCA
from utils import chebyshev_distance, euclidean_distance, grid, feature_standardize
import sys

sys.setrecursionlimit(1000000)


def load_samples(filename, standardize=False):
    with open(filename, 'r') as f:
        data = json.load(f)
        labels_real = np.array(data['labels_real'])
        labels_het = np.array(data['labels_het'])
        data = np.array(data['samples'])

    if standardize:
        data = (data - np.min(data)) / (np.max(data) - np.min(data))

    return data, labels_real, labels_het


def pca_show_figure(data, labels):
    mdl = PCA(n_components=2)
    projected_data = mdl.fit_transform(data)
    show_figure(projected_data, labels, title='PCA')


def show_figure(data, labels, alpha=0.6, title=''):
    plt.figure()
    plt.scatter(*data.T, c=labels, alpha=alpha)
    plt.title(title)
    plt.show()


def grouping(data, labels_r, labels_h, step_size, radius):
    # negative points
    data_n = data[labels_h == 0]
    labels_r_n = labels_r[labels_h == 0]
    n = data_n.shape[0]

    # Union-Find Set
    father = np.arange(0, n, 1, dtype=np.int)

    def find(x):
        if x == father[x]:
            return x
        else:
            father[x] = find(father[x])
            return father[x]

    for i in range(n):
        # print('')
        for j in range(i + 1, n):
            # print(chebyshev_distance(data_n[i], data_n[j]), end=' ')
            if chebyshev_distance(data_n[i], data_n[j]) <= step_size:
                father[find(i)] = find(j)

    groups = [[]]
    for i in range(n):
        for g in groups:
            if len(g) == 0:
                g.append(i)
                groups.append([])
                break
            elif find(g[0]) == find(i):
                g.append(i)
                break
    groups.pop()

    grouped_data = [[]] * (len(groups))
    grouped_labels_r = [[]] * (len(groups))
    grouped_labels_h = [[]] * (len(groups))

    for gid, g in enumerate(groups):
        grouped_data[gid] = data_n[g].tolist()
        grouped_labels_r[gid] = labels_r_n[g].tolist()
        grouped_labels_h[gid] = [0] * len(g)

    data_p = data[labels_h == 1]
    labels_r_p = labels_r[labels_h == 1]
    for features, label in zip(data_p, labels_r_p):
        min_gid = -1
        min_dist = radius
        for gid, g in enumerate(groups):
            for idx in g:
                dist = euclidean_distance(features, data_n[idx])
                if min_dist > dist:
                    min_dist, min_gid = dist, gid
        if min_gid != -1:
            grouped_data[min_gid].append(features)
            grouped_labels_r[min_gid].append(label)
            grouped_labels_h[min_gid].append(1)

    return grouped_data, grouped_labels_r, grouped_labels_h


def get_pca_result(step_size, radius):
    data, labels_real, labels_het = load_samples(os.path.join(settings.DATA_DIR, 'samples.json'), standardize=True)
    # print('Samples loaded.')
    n = data.shape[0]
    for i in range(n):
        data[i] = grid(data[i], n=10)
    # print('Data preprocessed.')

    # pca_show_figure(data, labels)

    # print('Grouping')
    grouped_data, grouped_labels_r, grouped_labels_h = grouping(data, labels_real, labels_het, step_size, radius)
    # g_size = [len(g) for g in grouped_labels]
    # print('Divided into {} Groups: ({})'.format(len(grouped_labels), g_size))

    hetero_list = []
    mdl = CPCA()
    for gid, (data, labels_r, labels_h) in enumerate(zip(grouped_data, grouped_labels_r, grouped_labels_h)):
        data = np.array(data)
        # print('Data: {}'.format(data))
        # print('Labels: {}'.format(labels))
        projected_data, cp = mdl.fit_transform(data, labels_h, alpha=0, standardized=False)
        projected_data[:, 0] = feature_standardize(projected_data[:, 0])
        projected_data[:, 1] = feature_standardize(projected_data[:, 1])
        try:
            projected_data = np.floor(projected_data * 10).astype(np.int)
        except TypeError:
            # print('Complex occurs. gid: {}, g_size: {}'.format(gid, len(labels)))
            projected_data = projected_data.real
            cp = cp.real
            projected_data = np.floor(projected_data * 10).astype(np.int)
        hetero_size = (np.array(labels_h) == 0).sum()
        count = np.zeros((2, 11, 11), dtype=np.int)
        for (d, label) in zip(projected_data, labels_r):
            count[label][d[0]][d[1]] += 1
        mat = np.zeros((11, 11), dtype=np.float)
        for i in range(11):
            for j in range(11):
                if count[0][i][j] == 0:
                    mat[i][j] = 0
                else:
                    mat[i][j] = count[0][i][j] / (count[0][i][j] + count[1][i][j])

        het = {
            'cp1': cp[0].tolist(),
            'cp2': cp[1].tolist(),
            'heteroSize': int(hetero_size),
            'dataMatrix': mat.tolist(),
        }
        hetero_list.append(het)
    return hetero_list
