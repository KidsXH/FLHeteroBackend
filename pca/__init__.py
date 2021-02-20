import json
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt

from FLHeteroBackend import settings
from pca.cpca import CPCA
from utils import chebyshev_distance, euclidean_distance, grid, feature_standardize


def load_samples(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
        outputs_server = np.array(data['outputs_server'])
        outputs_client = np.array(data['outputs_client'])
        ground_truth = np.array(data['ground_truth'])
        local_data = np.array(data['local_data'])
        samples = np.array(data['samples'])
    return samples, local_data, ground_truth, outputs_client, outputs_server


def pca_show_figure(data, labels, title='PCA'):
    mdl = PCA(n_components=2)
    projected_data = mdl.fit_transform(data)
    show_figure(projected_data, labels, title=title)


def show_figure(data, labels, alpha=0.6, title=''):
    plt.figure()
    plt.scatter(*data.T, c=labels, alpha=alpha)
    plt.title(title)
    plt.colorbar()
    plt.show()


def get_pca_result(n_clusters, data, ground_truth, outputs_client, outputs_server):
    n = data.shape[0]
    # PCA
    mdl = PCA(n_components=2)
    mdl.fit(data)
    pca = {
        'pc1': mdl.components_[0].tolist(),
        'pc2': mdl.components_[1].tolist(),
        'projectedData': mdl.transform(data).tolist(),
    }

    # Hetero
    idx = outputs_client != outputs_server
    hetero_samples = {'data': data[idx], 'labels': ground_truth[idx], 'index': np.arange(0, n, 1)[idx]}
    # Homo
    idx = outputs_client == outputs_server
    homo_samples = {'data': data[idx], 'labels': ground_truth[idx], 'index': np.arange(0, n, 1)[idx]}

    ac = AgglomerativeClustering(n_clusters=n_clusters)
    labels_cls = ac.fit_predict(hetero_samples['data'])
    labels_cls = labels_cls

    mdl = CPCA()
    hetero_list = []

    for i in range(n_clusters):
        idx = labels_cls == i
        bg = hetero_samples['data'][idx]
        fg = np.concatenate((homo_samples['data'], bg))

        transformed_data, cpcs = mdl.fit_transform(fg, bg, alpha=30)
        transformed_data = transformed_data.real
        cpcs = cpcs.real

        hetero_size = idx.sum()
        data_id = np.concatenate((homo_samples['index'], hetero_samples['index'][idx]))

        het = {
            'cpca': {
                'cpc1': cpcs[0].tolist(),
                'cpc2': cpcs[1].tolist(),
                'projectedData': transformed_data.tolist()
            },
            'heteroSize': int(hetero_size),
            'dataID': data_id.tolist()
        }
        hetero_list.append(het)

        # m = homo_samples['data'].shape[0]
        # plt.title('CLuster{}, alpha={}'.format(i, 0))
        # plt.scatter(*transformed_data[:m].T, c='black', alpha=0.3)
        # plt.scatter(*transformed_data[m:].T, c='red', alpha=0.3)
        # plt.show()

    return hetero_list, pca
