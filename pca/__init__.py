import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

from FLHeteroBackend import settings
from pca.cpca import CPCA
from pca.cluster import HeteroHierarchicalTree


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


def get_cluster_list(n_clusters, data, ground_truth, outputs_client, outputs_server):
    n = data.shape[0]

    # Hetero
    idx = outputs_client != outputs_server
    hetero_samples = {'data': data[idx], 'labels': ground_truth[idx], 'index': np.arange(0, n, 1)[idx]}
    # Homo
    idx = outputs_client == outputs_server
    homo_samples = {'data': data[idx], 'labels': ground_truth[idx], 'index': np.arange(0, n, 1)[idx]}

    tree = HeteroHierarchicalTree()
    tree.fit(outputs_client != outputs_server)
    cluster_rank, hetero_rates = tree.rank(hetero_samples['data'], n_clusters)
    cluster_labels = tree.labels  # type: np.ndarray
    hetero_list = []

    for i, ci in enumerate(cluster_rank):
        idx = cluster_labels == ci
        hetero_size = idx.sum()
        data_idx = hetero_samples['index'][idx]
        # bg = hetero_samples['data'][idx]
        # fg = np.concatenate((homo_samples['data'], bg))

        # transformed_data, cpcs = mdl.fit_transform(fg, bg, alpha=settings.DEFAULT_ALPHA)
        # transformed_data = transformed_data.real
        # cpcs = cpcs.real

        het = {
            # 'cpca': {
            #     'cpc1': cpcs[0].tolist(),
            #     'cpc2': cpcs[1].tolist(),
            # },
            'heteroSize': int(hetero_size),
            'heteroIndex': data_idx.tolist(),
            'heteroRate': hetero_rates[ci],
        }
        hetero_list.append(het)

        # if i < show_cpca_figures:
        #     m = homo_samples['data'].shape[0]
        #     plt.title('Cluster{}, Alpha={}, Het_rate={:.2f}%'.format(ci, settings.DEFAULT_ALPHA, hetero_rates[ci] * 100))
        #     plt.scatter(*transformed_data[:m].T, c='black', alpha=0.3)
        #     plt.scatter(*transformed_data[m:].T, c='red', alpha=0.3)
        #     plt.show()

    return hetero_list
