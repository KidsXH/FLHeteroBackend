import os

import numpy as np

from FLHeteroBackend import settings
from pca import build_tree
from pca.cluster import create_affinity


def preprocess(dataset, client_names, sampling_types):
    samples_file = os.path.join(settings.DATA_HOME[dataset], 'samples.npz')
    samples = np.load(samples_file)

    client_idx = []
    for idx, client_name in enumerate(samples['client_names']):
        if client_name in client_names:
            client_idx.append(idx)
    client_idx = np.array(client_idx)

    data = {'client_names': client_names, 'ground_truth': samples['ground_truth'][client_idx]}

    for sampling_type in sampling_types:
        data[sampling_type] = samples[sampling_type][client_idx]

    build_tree(data, sampling_types)
    create_affinity(data, sampling_types)


if __name__ == '__main__':
    preprocess(dataset='mnist_mlp', client_names=['Client-0', 'Client-2'], sampling_types=['local', 'stratified'])

    # samples_file = os.path.join(settings.DATA_HOME['mnist'], 'samples.npz')
    # samples = np.load(samples_file)
    # gt = samples['ground_truth']
    #
    # for idx, gt in enumerate(gt):
    #     print('Client-{}: {}'.format(idx, np.unique(gt).tolist()))
