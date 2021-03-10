import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

from FLHeteroBackend import settings
from pca.cpca import CPCA
from pca.cluster import HeteroHierarchicalTree, build_tree


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


def get_pcs(data):
    pca = PCA(n_components=2)
    pca.fit_transform(data)
    return pca.components_


def pca_weights(weights_0, weights_client, weights_server):
    pca = PCA(n_components=2)
    ws = pca.fit_transform(weights_server)
    wc = pca.transform(weights_client)
    w0 = pca.transform([weights_0])[0]
    return w0, wc, ws


def get_cluster_list(n_clusters, client_name, data, sampling_type, outputs_client, outputs_server):
    n = data.shape[0]

    # Hetero
    idx = outputs_client != outputs_server
    hetero_samples = {'data': data[idx], 'index': np.arange(0, n, 1)[idx]}
    # Homo
    idx = outputs_client == outputs_server
    homo_samples = {'data': data[idx], 'index': np.arange(0, n, 1)[idx]}
    # build_tree(data)
    tree = HeteroHierarchicalTree()
    tree.fit(client_name=client_name, sampling_type=sampling_type, hetero_labels=outputs_client != outputs_server)

    if n_clusters is None:
        _, _ = tree.rank(n_clusters=2)
        distance_rev = tree.distances[:][::-1]
        acceleration_rev = np.diff(distance_rev, 2)
        n_clusters = acceleration_rev.argmax() + 2

    cluster_rank, hetero_rates = tree.rank(n_clusters)
    cluster_labels = tree.labels  # type: np.ndarray

    hetero_list = []

    for i, ci in enumerate(cluster_rank):
        idx = cluster_labels == ci
        hetero_size = idx.sum()
        data_idx = hetero_samples['index'][idx]

        het = {
            'heteroSize': int(hetero_size),
            'heteroIndex': data_idx.tolist(),
            'heteroRate': hetero_rates[ci],
        }
        hetero_list.append(het)

    return hetero_list
