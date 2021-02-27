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
        self.hetero_labels = None
        self.n_leaves = None
        self.het_count = None
        self.all_count = None
        self.father = None
        self.depth = None
        self.labels = None
        self.fitted = False

    def fit(self, hetero_labels):
        self.children = children = load_children()
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

    # def split(self, min_het_rate):
    #     n_leaves, het_count, all_count, children = self.n_leaves, self.het_count, self.all_count, self.children
    #     queue = Queue()
    #     queue.put(n_leaves * 2 - 2)
    #     cluster_roots = []
    #
    #     while not queue.empty():
    #         cur_node = queue.get()
    #         het_rate = het_count[cur_node] / all_count[cur_node]
    #         if het_rate >= min_het_rate:
    #             cluster_roots.append(cur_node)
    #             continue
    #         if het_count[cur_node] > 0:
    #             queue.put(children[cur_node - n_leaves][0])
    #             queue.put(children[cur_node - n_leaves][1])
    #
    #     cls_labels = np.zeros(n_leaves, dtype=int)
    #     label_count = 1
    #     for root in cluster_roots:
    #         if all_count[root] < 5:
    #             continue
    #         lv_idx = _find_leaves_in_subtree(children, n_leaves, root)
    #         cls_labels[lv_idx] = label_count
    #         label_count += 1
    #     return cls_labels

    def rank(self, hetero_data, n_clusters):
        if not self.fitted:
            raise ValueError('Not fitted yet.')
        hetero_labels, n_leaves, children, father, depth = self.hetero_labels, self.n_leaves, self.children, self.father, self.depth
        idx_het = np.arange(0, n_leaves, 1, dtype=int)[hetero_labels]
        ac = AgglomerativeClustering(n_clusters=n_clusters)
        cls_labels_het = ac.fit_predict(hetero_data)
        self.labels = cls_labels_het

        roots = []
        hetero_rates = []
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
            hetero_rates.append(self.get_het_rate(root))
        rank = np.argsort(roots)
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


def build_tree(data):
    children, n_components, n_leaves, parent = ward_tree(data)
    np.savez_compressed(settings.TREE_FILE, children=children)


def load_children():
    data = np.load(settings.TREE_FILE)
    return data['children']


def clustering_history(weight_0, weights_server, weights_client, n_clusters=20):
    pca = PCA(n_components=2)
    transformed_weight_s = pca.fit_transform(weights_server)
    transformed_weight_c = pca.transform(weights_client)
    transformed_weight_0 = pca.transform([weight_0])[0]
    segmentations = temp_segment(transformed_weight_s, n_clusters)

    end_points = np.array([s[0] for s in segmentations] + [segmentations[-1][-1]])
    # clustered_weights_s = transformed_weight_s[end_points]
    # clustered_weights_c = transformed_weight_c[end_points]

    # return transformed_weight_0, clustered_weights_s, clustered_weights_c, end_points
    return transformed_weight_0, transformed_weight_s, transformed_weight_c, end_points


def temp_segment(data, n_segments=20):
    T, n = data.shape

    if T < n_segments:
        return data

    n = data.shape[0]
    f = np.zeros(n, dtype=float)
    for i in range(1, n):
        f[i] = np.linalg.norm(data[i] - data[i - 1])

    low, high = 0, f.sum()
    step = 0
    while low < high:
        step += 1
        mid = (low + high) / 2
        result = [[0]]
        sum_length = 0
        for i in range(1, T):
            d = np.linalg.norm(data[i] - data[i - 1])
            if mid >= sum_length + d:
                result[-1].append(i)
                sum_length += d
            else:
                result.append([i])
                sum_length = d

        len_r = len(result)

        if len_r == n_segments or step == 100:
            # print(mid)
            return result
        elif len_r > n_segments:
            low = mid
        else:
            high = mid
