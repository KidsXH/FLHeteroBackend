import os
from time import time

import numpy as np
from sklearn.cluster import ward_tree
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

from FLHeteroBackend import settings
from pca.cluster import HeteroHierarchicalTree, build_tree
from pca import pca_show_figure
from utils import load_samples, load_weights

tree_file = os.path.join(settings.DATA_DIR, 'tree.npz')


def test_1():
    samples, local_data, ground_truth, outputs_client, outputs_server = load_samples(settings.SAMPLE_FILE)

    data_het = samples  # [outputs_client != outputs_server]
    print('Number of hetero points: {}'.format(data_het.shape[0]))
    start_time = time()
    children, n_components, n_leaves, parent = ward_tree(data_het)
    np.savez_compressed(tree_file, children=children)
    print(children.shape)
    end_time = time()
    print('Time Consuming: {}'.format(end_time - start_time))


def test_3(n_clusters):
    samples, local_data, ground_truth, outputs_client, outputs_server = load_samples(settings.SAMPLE_FILE)
    hetero_labels = outputs_server != outputs_client
    hetero_data = samples[hetero_labels]
    tree = HeteroHierarchicalTree()
    tree.fit(hetero_labels)
    cluster_rank, hetero_rates = tree.rank(hetero_data, n_clusters)
    # pca_show_figure(hetero_data, tree.labels)
    print(cluster_rank)
    print(hetero_rates[cluster_rank].astype(float))


def test_4():
    samples, local_data, ground_truth, outputs_client, outputs_server = load_samples(settings.SAMPLE_FILE)
    get_pca_result(5, samples, ground_truth, outputs_client, outputs_server)


if __name__ == '__main__':
    # samples, local_data, ground_truth, outputs_client, outputs_server = load_samples(settings.SAMPLE_FILE)
    # print(samples.shape)
    # print(np.unique(samples, axis=0).shape)
    # build_tree(samples)
    # test_1()
    # test_4()
    #
    # start_time = time()
    # weights = np.random.rand(200, 7850)
    # print(weights.shape)
    # tsne = TSNE()
    # X = tsne.fit_transform(weights)
    # print(X.shape)
    # pca = PCA(n_components=2)
    # X = pca.fit_transform(weights)
    # print(X.shape)
    # print('Time: {}'.format(time() - start_time))
    weights = load_weights('mnist', 'Client-0', 200)
    w0 = weights['weights_0']
    weights_server = weights['weights_server']
    weights_client_0 = weights['weights_client']
    weights_client_1 = load_weights('mnist', 'Client-9', 200)['weights_client']

    m = weights_server.shape[0]

    pca = PCA(n_components=2)
    # X = pca.fit_transform(np.concatenate((weights_server, weights_client)))
    X = pca.fit_transform(weights_server)

    Y = pca.transform(weights_client_0)
    Z = pca.transform(weights_client_1)
    w0 = pca.transform([w0])
    print(w0)
    # Y = X[m:]
    # X = X[:m]

    # tsne = TSNE()
    # X = tsne.fit_transform(weights_server)

    plt.figure()
    plt.scatter(*w0.T, c='red', alpha=0.3)
    plt.scatter(*Y.T, c='black', alpha=0.3)
    # plt.scatter(*Z.T, c='red', alpha=0.3)
    plt.plot(*X.T, c='blue', alpha=0.6)
    plt.title('PCA')
    plt.show()
