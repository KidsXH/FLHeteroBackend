import os
from time import time

import numpy as np

from FLHeteroBackend import settings
from pca import build_tree
from pca.cluster import create_affinity


def preprocess(dataset, client_names, sampling_types):
    build_tree(dataset, client_names, sampling_types)
    create_affinity(dataset, client_names, sampling_types)


if __name__ == '__main__':
    start_time = time()
    preprocess(dataset='mnist', client_names=['Client-0', 'Client-2'], sampling_types=['local', 'stratified'])
    print('Finish in', time() - start_time)
    # samples_file = os.path.join(settings.DATA_HOME['mnist'], 'samples.npz')
    # samples = np.load(samples_file)
    # gt = samples['ground_truth']
    #
    # for idx, gt in enumerate(gt):
    #     print('Client-{}: {}'.format(idx, np.unique(gt).tolist()))
