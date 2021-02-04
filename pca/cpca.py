import numpy as np
from numpy import linalg as LA
from functools import reduce


class CPCA(object):
    DEFAULT_ALPHAS = np.concatenate(([0], np.logspace(-1, 3, 99)))

    def get_labels(self):
        return self.labels

    def set_labels(self, labels):
        if np.shape(labels)[0] != self.fg_n:
            raise ValueError('The number of data and labels must be the same')
        self.labels = np.array(labels)

    def is_fitted(self):
        return self.fitted

    def __init__(self, n_components=2):
        self.n_components = n_components
        self.fitted = False
        self.fg = None
        self.bg = None
        self.fg_n, self.fg_d = None, None
        self.bg_n, self.bg_d = None, None
        self.fg_cov = None
        self.bg_cov = None
        self.labels = None  # {bg} = 1 {fg}/{bg} = 0

    def fit_transform(self, dataset, labels, standardized=True,
                      alpha=0):
        self.fit(dataset, labels, standardized)
        return self.transform(alpha)

    def fit(self, dataset, labels, standardized=True):
        self.fg = np.array(dataset)
        self.fg_n, self.fg_d = self.fg.shape

        self.set_labels(labels)

        self.bg = np.array(self.fg[self.labels == 1])
        self.bg_n, self.bg_d = self.bg.shape

        # center the data
        self.fg = self.fg - np.mean(self.fg, axis=0)
        self.bg = self.bg - np.mean(self.bg, axis=0)

        # standardize
        if standardized:
            self.fg = np.nan_to_num(self.fg / np.std(self.fg, axis=0))
            self.bg = np.nan_to_num(self.bg / np.std(self.bg, axis=0))

        # calculate the covariance matrices
        self.fg_cov = self.fg.T.dot(self.fg) / (self.fg_n - 1)
        self.bg_cov = self.bg.T.dot(self.bg) / (self.bg_n - 1)

        self.fitted = True

    def transform(self, alpha=0):
        if not self.fitted:
            raise ValueError('This model has not been fit to a foreground/background dataset yet. Please run the fit('
                             ') or fit_transform() functions first.')

        transformed_data, cp = self.cpca(alpha)
        return transformed_data, cp

    def cpca(self, alpha):
        if not self.fitted:
            raise ValueError('This model has not been fit to a foreground/background dataset yet. Please run the fit('
                             ') or fit_transform() functions first.')
        sigma = self.fg_cov - alpha * self.bg_cov
        # sigma = sigma.astype(np.float128)
        # print('SUM: {}'.format(np.sum(sigma - sigma.T)))
        w, v = LA.eig(sigma)
        eig_idx = np.argpartition(w, -self.n_components)[-self.n_components:]
        eig_idx = eig_idx[np.argsort(-w[eig_idx])]
        v_top = v[:, eig_idx]
        projected_data = self.fg.dot(v_top)
        return projected_data, v_top.T
