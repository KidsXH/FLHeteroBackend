import os
from queue import Queue

import numpy as np
from sklearn.cluster import ward_tree, AgglomerativeClustering
from sklearn.decomposition import PCA

from FLHeteroBackend import settings


class HeteroHierarchicalTree:
    """
    Pre-process: build_tree(samples)
    Process:
        tree = HeteroHierarchicalTree()
        tree.fit(hetero_labels)
        cluster_rank = tree.rank(hetero_data, n_clusters)
    """
    def __init__(self):
        self.children = None
        self.affinity = None
        self.hetero_labels = None
        self.n_leaves = None
        self.het_count = None
        self.all_count = None
        self.father = None
        self.depth = None
        self.labels = None
        self.distances = None
        self.fitted = False

    def fit(self, dataset, client_name, sampling_type, hetero_labels):
        self.children = children = load_children(dataset, client_name, sampling_type)
        self.affinity = load_affinity(dataset, client_name, sampling_type)
        self.hetero_labels = hetero_labels
        self.n_leaves = n_leaves = children.shape[0] + 1

        all_idx = np.arange(0, n_leaves, 1)
        all_count = np.zeros(n_leaves * 2 - 1, dtype=int)
        all_count[all_idx] += 1

        het_idx = all_idx[hetero_labels]
        het_count = np.zeros(n_leaves * 2 - 1, dtype=int)
        het_count[het_idx] += 1

        father = np.zeros(n_leaves * 2 - 1, dtype=int)

        for i, child in enumerate(children):
            het_count[n_leaves + i] = het_count[child[0]] + het_count[child[1]]
            all_count[n_leaves + i] = all_count[child[0]] + all_count[child[1]]
            father[child[0]] = n_leaves + i
            father[child[1]] = n_leaves + i

        self.het_count = het_count
        self.all_count = all_count
        self.father = father
        self.depth = _get_depth(children, n_leaves)

        self.fitted = True

    def rank(self, n_clusters):
        if not self.fitted:
            raise ValueError('Not fitted yet.')
        hetero_labels, n_leaves, children, father, depth = self.hetero_labels, self.n_leaves, self.children, self.father, self.depth
        idx_het = np.arange(0, n_leaves, 1, dtype=int)[hetero_labels]

        affinity = self.affinity[hetero_labels][:, hetero_labels]
        cls = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='single',
                                      compute_distances=True)
        cls_labels_het = cls.fit_predict(affinity)
        self.labels = cls_labels_het
        self.distances = cls.distances_

        roots = []
        hetero_rates = []
        hetero_size = []
        for ci in range(n_clusters):
            # LCA (can be optimized)
            nodes = idx_het[cls_labels_het == ci]
            cur_depth = np.min(depth[nodes])
            n_nodes = nodes.shape[0]
            for i in range(n_nodes):
                while depth[nodes[i]] > cur_depth:
                    nodes[i] = father[nodes[i]]
            nodes = np.unique(nodes)
            while nodes.shape[0] > 1:
                for i, node in enumerate(nodes):
                    nodes[i] = father[nodes[i]]
                nodes = np.unique(nodes)
            root = nodes[0]
            # print('Cluster {}: Root={}, Depth={}, Het_Count/All_Count={}/{}'.format(ci, root, depth[root],
            #                                                                         self.het_count[root],
            #                                                                         self.all_count[root],))
            roots.append(root)
            hetero_rates.append(n_nodes / self.all_count[root])
            hetero_size.append(n_nodes)
        rank = np.argsort(hetero_size)[::-1]
        hetero_rates = np.array(hetero_rates).astype(float)
        return rank, hetero_rates

    def get_het_rate(self, node):
        return self.het_count[node] / self.all_count[node]


def _find_leaves_in_subtree(children, n_leaves, root):
    lv_list = []
    queue = Queue()
    queue.put(root)
    while not queue.empty():
        cur_node = queue.get()
        if cur_node < n_leaves:
            lv_list.append(cur_node)
        else:
            queue.put(children[cur_node - n_leaves][0])
            queue.put(children[cur_node - n_leaves][1])
    return np.array(lv_list)


def _get_depth(children, n_leaves):
    depth = np.zeros(n_leaves * 2 - 1, dtype=int)
    queue = Queue()
    queue.put(depth.shape[0] - 1)
    while not queue.empty():
        cur_node = queue.get()
        if cur_node >= n_leaves:
            depth[children[cur_node - n_leaves][0]] = depth[cur_node] + 1
            depth[children[cur_node - n_leaves][1]] = depth[cur_node] + 1
            queue.put(children[cur_node - n_leaves][0])
            queue.put(children[cur_node - n_leaves][1])
    return depth


def build_tree(dataset, client_list, sampling_types):
    samples_data = np.load(os.path.join(settings.DATA_HOME[dataset], 'samples.npz'), allow_pickle=True)
    client_names = samples_data['client_names']
    trees = {}
    for client_idx, client_name in enumerate(client_names):
        if client_name not in client_list:
            continue
        print('Building Tree: {}'.format(client_name))
        trees[client_name] = {}
        for sampling_type in sampling_types:
            print('   Data Shape:', samples_data[sampling_type][client_idx].shape)
            children, n_components, n_leaves, parent = ward_tree(samples_data[sampling_type][client_idx])
            trees[client_name][sampling_type] = children
    tree_file = os.path.join(settings.CACHE_DIR, 'tree_{}'.format(dataset))
    np.savez_compressed(tree_file, **trees)


def load_children(dataset, client_name, sampling_type):
    tree_file = os.path.join(settings.CACHE_DIR, 'tree_{}.npz'.format(dataset))
    data = np.load(tree_file, allow_pickle=True)
    return data[client_name].item()[sampling_type]


def create_affinity(dataset, client_list, sampling_types):
    samples_data = np.load(os.path.join(settings.DATA_HOME[dataset], 'samples.npz'), allow_pickle=True)
    client_names = samples_data['client_names'][:2]
    affinity = {}
    for client_idx, client_name in enumerate(client_names):
        if client_name not in client_list:
            continue
        print('Creating Affinity: {}'.format(client_name))
        affinity[client_name] = {}
        for sampling_type in sampling_types:
            print('  Data Shape:', samples_data[sampling_type][client_idx].shape)
            affinity[client_name][sampling_type] = calculate_affinity(samples_data[sampling_type][client_idx])
    affinity_file = os.path.join(settings.CACHE_DIR, 'affinity_{}'.format(dataset))
    np.savez_compressed(affinity_file, **affinity)


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


def load_affinity(dataset, client_name, sampling_type):
    affinity_file = os.path.join(settings.CACHE_DIR, 'affinity_{}.npz'.format(dataset))
    data = np.load(affinity_file, allow_pickle=True)
    return data[client_name].item()[sampling_type]
