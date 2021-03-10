import json
import os
from time import time

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

from FLHeteroBackend import settings
from pca import CPCA, build_tree
from pca.cluster import create_affinity
from utils import load_outputs

data_file = os.path.join(settings.DATA_DIR, 'samples.npz')
output_file = os.path.join(settings.DATA_DIR, 'output.npz')
affinity_file = os.path.join(settings.CACHE_DIR, 'affinity.npz')


def test_cpca():
    data = np.load(data_file)['data']
    output = np.load(output_file)
    hetero_labels = output['output_server'] != output['output_client']
    cpca = CPCA(n_components=2)
    cpca.fit(target=data, background=data[hetero_labels == 1])
    cpca.find_best_alpha(n_candidates=5)
    X_0 = cpca.transform()
    alpha = cpca.best_alpha

    # plt.title('CPCA alpha=0')
    # plt.scatter(*X_0[hetero_labels == 0].T, c='black', alpha=0.6)
    # plt.scatter(*X_0[hetero_labels == 1].T, c='red', alpha=0.6)
    # plt.show()
    #
    # alpha = 50
    X_1 = cpca.transform(alpha=alpha)
    plt.title('CPCA alpha={}'.format(alpha))
    plt.scatter(*X_1[hetero_labels == 0].T, c='black', alpha=0.6)
    plt.scatter(*X_1[hetero_labels == 1].T, c='red', alpha=0.6)
    plt.show()

    X_1 = cpca.transform(alpha=alpha)
    plt.title('CPCA alpha={}'.format(alpha))
    plt.scatter(*X_1[hetero_labels == 1].T, c='red', alpha=0.6)
    plt.scatter(*X_1[hetero_labels == 0].T, c='black', alpha=0.6)
    plt.show()


def test_clustering(dataset, n_clusters, show_cluster, sample_type, client_name, cm_round):
    f_data = np.load(os.path.join(settings.DATA_HOME[dataset], 'samples.npz'))
    client_idx = int(client_name[-1])
    data = f_data[sample_type][client_idx]
    gt = f_data['ground_truth'][client_idx]
    n = data.shape[0]
    labels = np.zeros(n, dtype=int)
    labels_d = np.zeros(n, dtype=int)
    affinity = np.load(settings.AFFINITY_file, allow_pickle=True)[client_name].item()[sample_type]

    output = load_outputs(datasets=dataset, client_name=client_name, cm_round=cm_round, sampling_type=sample_type)
    hetero_labels = output['outputs_server'] != output['outputs_client']
    hetero_idx = np.arange(0, data.shape[0], 1, dtype=int)[hetero_labels]
    affinity = affinity[hetero_labels][:, hetero_labels]

    print('Total hetero points: {}'.format(np.sum(hetero_labels)))
    print('Server Output: {}'.format(np.unique(output['outputs_server']).tolist()))
    print('Client Output: {}'.format(np.unique(output['outputs_client']).tolist()))

    cls_dis = AgglomerativeClustering(n_clusters=n_clusters)
    cls_dis.fit_predict(data[hetero_labels])
    cls_dis_labels = cls_dis.labels_ + 1
    labels_d[hetero_idx] = cls_dis_labels

    cls = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='single')
    cls.fit_predict(affinity)
    cls_labels = cls.labels_ + 1

    labels[hetero_idx] = cls_labels

    print('Points in cluster: {}'.format(np.sum(labels == show_cluster)))

    cpca = CPCA(n_components=2)
    cpca.fit(target=data, background=data[labels == show_cluster])
    pca = PCA(n_components=2)
    X_0 = pca.fit_transform(data)
    # cpca.find_best_alpha(n_candidates=5)
    # print(np.where(X_0[:, 0] < -1000))
    plt.title('PCA Clusters (by rank)')

    for label in range(n_clusters + 1):
        plt.scatter(*X_0[labels == label].T, alpha=0.6)
        # print((cls_labels == label).sum())

    plt.show()

    plt.title('PCA Clusters (by distance)')

    for label in range(n_clusters + 1):
        plt.scatter(*X_0[labels_d == label].T, alpha=0.6)
        # print((cls_labels == label).sum())

    plt.show()
    #
    # plt.title('PCA Ground Truth')
    #
    # plt.scatter(*X_0[gt == 0].T, alpha=0.6)
    # plt.scatter(*X_0[gt == 1].T, alpha=0.6)
    #
    # plt.show()

    # print(np.where(X_0[:, 0] > 20))
    # print(np.where(X_0[:, 1] > 20))

    saved_data = {'data': np.array(X_0).tolist(),
                  'labels_by_rank': labels.tolist(),
                  'labels_by_dist': labels_d.tolist(),
                  }
    with open(os.path.join(settings.DATA_DIR, '{}_clusters.json'.format(client_name)), 'w') as f:
        json.dump(saved_data, f)
    #
    return
    # plt.title('PCA Server OP')
    #
    # plt.scatter(*X_0[output['outputs_server'] == 0].T, alpha=0.6)
    # plt.scatter(*X_0[output['outputs_server'] == 1].T, alpha=0.6)
    #
    # plt.show()
    #
    # plt.title('PCA Client OP')
    #
    # plt.scatter(*X_0[output['outputs_client'] == 0].T, alpha=0.6)
    # plt.scatter(*X_0[output['outputs_client'] == 1].T, alpha=0.6)
    #
    # plt.show()
    # cpca.find_best_alpha(n_candidates=5)
    alpha = 6
    X_1 = cpca.transform(alpha=alpha)
    plt.title('CPCA alpha={}'.format(alpha))
    plt.scatter(*X_1[labels == 0].T, c='black', alpha=0.6)
    plt.scatter(*X_1[labels == show_cluster].T, c='red', alpha=0.6)
    plt.show()
    plt.title('CPCA alpha={}'.format(alpha))
    plt.scatter(*X_1[labels == show_cluster].T, c='red', alpha=0.6)
    plt.scatter(*X_1[labels == 0].T, c='black', alpha=0.6)
    plt.show()

    cpca.find_best_alpha(n_candidates=5)
    alpha = cpca.best_alpha
    X_1 = cpca.transform(alpha=alpha)

    plt.title('CPCA alpha={}'.format(alpha))
    for label in range(n_clusters + 1):
        plt.scatter(*X_1[labels == label].T, alpha=0.6)
    plt.show()

    plt.title('CPCA alpha={}'.format(alpha))
    plt.scatter(*X_1[labels == show_cluster].T, c='red', alpha=0.6)
    plt.scatter(*X_1[labels == 0].T, c='black', alpha=0.6)
    plt.show()

    return

    cpc = cpca.components_
    X1 = np.matmul(data, cpc.T)
    plt.title('CPCA alpha={}'.format(alpha))
    for label in range(n_clusters + 1):
        plt.scatter(*X_1[labels == label].T, alpha=0.6)
    plt.show()

    return

    maxi = np.max(X1[:, 0])
    mini = np.min(X1[:, 0])
    xx = (X1[:, 0] - mini) / (maxi - mini)
    maxi = np.max(X1[:, 1])
    mini = np.min(X1[:, 1])
    yy = (X1[:, 1] - mini) / (maxi - mini)
    X_1 = np.array([xx, yy]).T
    plt.axes().set_aspect('equal')
    plt.title('CPCA alpha={}'.format(alpha))
    plt.scatter(*X_1[labels == 0].T, c='black', alpha=0.6)
    # plt.scatter(*X_1[(labels == show_cluster) & (gd == 6)].T, c='yellow', alpha=0.6)
    plt.scatter(*X_1[(labels == show_cluster) & (gd == 0)].T, c='red', alpha=0.6)
    plt.scatter(*X_1[labels == 2].T, c='blue', alpha=0.6)
    plt.show()


def calculate_affinity(data):
    n, d = data.shape
    affinity = np.zeros((n, n), np.float32)
    distances = np.zeros((n, n), np.float32)
    rank = np.zeros((n, n), np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            distances[i][j] = distances[j][i] = np.linalg.norm(data[i] - data[j])
        rank[i] = np.argsort(np.argsort(distances[i]))

    for i in range(n):
        for j in range(i + 1, n):
            affinity[i][j] = affinity[j][i] = rank[i][j] * rank[j][i]

    return affinity


if __name__ == '__main__':
    last_time = time()

    test_clustering(dataset='mnist_mlp', n_clusters=10, show_cluster=1, sample_type='stratified',
                    client_name='Client-0', cm_round=19)

    # data = np.load(settings.DATA_HOME['mnist'])

    # data = np.load(settings.TREE_FILE, allow_pickle=True)
    # print(data['Client-0'].item()['local'].shape)

    # build_tree(data, sampling_types=['local', 'stratified'])
    # create_affinity(data, sampling_types=['local', 'stratified'])
    print('Time: {}'.format(time() - last_time))
