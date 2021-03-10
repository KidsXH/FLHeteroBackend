from time import time

import numpy as np
from numpy import linalg as LA
from sklearn.cluster import SpectralClustering


class CPCA(object):
    DEFAULT_ALPHAS = np.concatenate(([0], np.logspace(-1, 3, 39)))

    def __init__(self, n_components=2):
        self.n_components = n_components
        self.fitted = False
        self.fg = None
        self.bg = None
        self.fg_n, self.fg_d = None, None
        self.bg_n, self.bg_d = None, None
        self.fg_cov = None
        self.bg_cov = None
        self.alpha = None
        self.best_alpha = None

        self.components_ = None

    def fit_transform(self, target, background, alpha=None):
        self.fit(target, background)
        return self.transform(alpha)

    def fit(self, target, background):
        self.fg = np.array(target)
        self.fg_n, self.fg_d = self.fg.shape

        self.bg = np.array(background)
        self.bg_n, self.bg_d = self.bg.shape
        # center the data
        self.fg = self.fg - np.mean(self.fg, axis=0)
        self.bg = self.bg - np.mean(self.bg, axis=0)

        # calculate the covariance matrices
        self.fg_cov = self.fg.T.dot(self.fg) / (self.fg_n - 1)
        self.bg_cov = self.bg.T.dot(self.bg) / (self.bg_n - 1)

        self.fitted = True

    def transform(self, alpha=None):
        if not self.fitted:
            raise ValueError('Not fitted yet.')

        if alpha is None:
            if self.best_alpha is None:
                self.find_best_alpha()
            alpha = self.best_alpha

        sigma = self.fg_cov - alpha * self.bg_cov
        w, v = LA.eig(sigma)
        eig_idx = np.argpartition(w, -self.n_components)[-self.n_components:]
        eig_idx = eig_idx[np.argsort(-w[eig_idx])]
        v_top = v[:, eig_idx]
        projected_data = self.fg.dot(v_top)

        self.components_ = v_top.T.real
        self.alpha = alpha

        return projected_data.real

    def find_best_alpha(self, alphas=None, n_candidates=5):
        if alphas is None:
            alphas = self.DEFAULT_ALPHAS

        affinity = self.create_affinity_matrix(alphas)
        # print(affinity.shape)

        spectral = SpectralClustering(n_clusters=n_candidates, affinity='precomputed')

        spectral.fit(affinity)
        labels = spectral.labels_

        # print(labels)

        # we see middle candidate as the best one

        first_idx = np.sort([np.where(labels == label)[0][0] for label in np.unique(labels)])
        selected_label = labels[first_idx[n_candidates // 2]]
        # print(first_idx)
        # print(selected_label)

        idx = np.where(labels == selected_label)[0]
        affinity_sub_matrix = affinity[idx][:, idx]
        sum_affinities = np.sum(affinity_sub_matrix, axis=0)
        exemplar_idx = idx[np.argmax(sum_affinities)]
        best_alpha = alphas[exemplar_idx]

        self.best_alpha = best_alpha
        return best_alpha

    def create_affinity_matrix(self, alphas):
        subspaces = list()
        k = len(alphas)
        affinity = 0.5 * np.identity(k)

        last_time = time()

        for alpha in alphas:
            space = self.transform(alpha=alpha)
            q, r = np.linalg.qr(space)
            subspaces.append(q)

        print('create_affinity_matrix finished in {}s'.format(time() - last_time))

        for i in range(k):
            for j in range(i + 1, k):
                q0 = subspaces[i]
                q1 = subspaces[j]
                u, s, v = np.linalg.svd(q0.T.dot(q1))
                affinity[i, j] = s[0] * s[1]
        affinity = affinity + affinity.T
        affinity_matrix = np.nan_to_num(affinity)
        return affinity_matrix

