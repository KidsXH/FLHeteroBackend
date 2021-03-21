import os
from time import time

import numpy as np

from FLHeteroBackend import settings
from cluster import create_affinity


def preprocess(dataset, client_names):
    create_affinity(dataset, client_names)


if __name__ == '__main__':
    start_time = time()
    preprocess(dataset='cifar10', client_names=['Client-0', 'Client-1', 'Client-2', 'Client-3'])
    # preprocess(dataset='face', client_names=['Client-0', 'Client-1'])
    # preprocess(dataset='cifar10', client_names=['Client-2', 'Client-7'])
    print('Finish in', time() - start_time)
    # samples_file = os.path.join(settings.DATA_HOME['mnist'], 'samples.npz')
    # samples = np.load(samples_file)
    # gt = samples['ground_truth']
    #
    # for idx, gt in enumerate(gt):
    #     print('Client-{}: {}'.format(idx, np.unique(gt).tolist()))
