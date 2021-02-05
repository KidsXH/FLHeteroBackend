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
        fed_result = np.array(data['fed_result'])
        data = np.array(data['samples'])

    if standardize:
        data = (data - np.min(data)) / (np.max(data) - np.min(data))

    return data, labels_real, labels_het, fed_result


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
    neg_idx = labels_h == 0
    data_n = data[neg_idx]
    labels_r_n = labels_r[neg_idx]
    id_n = np.arange(0, data.shape[0], 1, dtype=int)[neg_idx]
    n = neg_idx.sum()

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

    grouped_id = [[]] * (len(groups))
    grouped_data = [[]] * (len(groups))
    grouped_labels_r = [[]] * (len(groups))
    grouped_labels_h = [[]] * (len(groups))

    for gid, g in enumerate(groups):
        grouped_id[gid] = id_n[g].tolist()
        grouped_data[gid] = data_n[g].tolist()
        grouped_labels_r[gid] = labels_r_n[g].tolist()
        grouped_labels_h[gid] = [0] * len(g)

    for idx, (features, label) in enumerate(zip(data, labels_r)):
        if labels_h[idx] == 0:
            continue
        min_gid = -1
        min_dist = radius
        for gid, g in enumerate(groups):
            for _idx in g:
                dist = euclidean_distance(features, data_n[_idx])
                if min_dist > dist:
                    min_dist, min_gid = dist, gid
        if min_gid != -1:
            grouped_id[min_gid].append(idx)
            grouped_data[min_gid].append(features)
            grouped_labels_r[min_gid].append(label)
            grouped_labels_h[min_gid].append(1)

    return grouped_id, grouped_data, grouped_labels_r, grouped_labels_h


def get_pca_result(step_size, radius):
    data, labels_real, labels_het, fed_result = load_samples(os.path.join(settings.DATA_DIR, 'samples.json'),
                                                             standardize=True)
    # print('Samples loaded.')
    n = data.shape[0]
    for i in range(n):
        data[i] = grid(data[i], n=10)
    # print('Data preprocessed.')

    # pca_show_figure(data, labels)

    # print('Grouping')
    grouped_id, grouped_data, grouped_labels_r, grouped_labels_h = grouping(data, labels_real, labels_het, step_size,
                                                                            radius)
    # g_size = [len(g) for g in grouped_labels]
    # print('Divided into {} Groups: ({})'.format(len(grouped_labels), g_size))
    # print(grouped_id)
    mdl = PCA()
    mdl.fit(data)
    pca = {
        'cp1': mdl.components_[0].tolist(),
        'cp2': mdl.components_[1].tolist(),
        'projectedData': mdl.transform(data).tolist(),
    }

    hetero_list = []
    mdl = CPCA()
    for gid, (idx, data, labels_r, labels_h) in enumerate(zip(grouped_id, grouped_data, grouped_labels_r,
                                                              grouped_labels_h)):
        data = np.array(data)
        # print('Data: {}'.format(data))
        # print('Labels: {}'.format(labels))
        projected_data, cp = mdl.fit_transform(data, labels_h, alpha=0, standardized=False)
        c_data = projected_data
        c_data[:, 0] = feature_standardize(projected_data[:, 0])
        c_data[:, 1] = feature_standardize(projected_data[:, 1])
        try:
            c_data = np.floor(c_data * 10).astype(np.int)
        except TypeError:
            print('Complex transferred to real. gid: {}, g_size: {}'.format(gid, len(labels_r)))
            projected_data = projected_data.real
            c_data = c_data.real
            cp = cp.real
            c_data = np.floor(c_data * 10).astype(np.int)
        hetero_size = (np.array(labels_h) == 0).sum()
        count = np.zeros((2, 11, 11), dtype=np.int)
        for (d, label) in zip(c_data, labels_r):
            count[label][d[0]][d[1]] += 1
        mat = np.zeros((11, 11), dtype=np.float)
        for i in range(11):
            for j in range(11):
                if count[0][i][j] == 0:
                    mat[i][j] = 0
                else:
                    mat[i][j] = count[0][i][j] / (count[0][i][j] + count[1][i][j])

        het = {
            'cpca': {
                'cp1': cp[0].tolist(),
                'cp2': cp[1].tolist(),
                'projectedData': projected_data.tolist()
            },
            'heteroSize': int(hetero_size),
            'dataMatrix': mat.tolist(),
            'dataID': idx
        }
        hetero_list.append(het)
    return hetero_list, pca, labels_het.tolist(), fed_result.tolist()
