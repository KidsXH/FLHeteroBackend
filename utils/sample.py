import os

import numpy as np

from FLHeteroBackend import settings
from fedlearn.datasets.mnist import get_mnist_data
from utils import grid_by_value, grid_by_percentile


def sampling(datasets='mnist'):
    client_names, train_loaders, test_loaders, val_loaders = get_mnist_data()

    data_home = settings.DATA_HOME[datasets]

    for client_name, val_loader in zip(client_names, val_loaders):
        print('Sampling for {}.'.format(client_name))
        samples_data_file = os.path.join(data_home, 'samples', '{}_samples.npz'.format(client_name))

        local_data = []
        ground_truth = []

        for (feats, labels) in val_loader:
            local_data.append(feats.float().numpy())
            ground_truth.append(labels.long().numpy())

        local_data = np.concatenate(local_data)
        ground_truth = np.concatenate(ground_truth)

        stratified_data = [grid_by_value(x) for x in local_data]
        systematic_data = [grid_by_percentile(x) for x in local_data]

        np.savez(samples_data_file, local_data=local_data, stratified_data=stratified_data,
                 systematic_data=systematic_data, ground_truth=ground_truth)


if __name__ == '__main__':
    sampling()
    data = np.load(os.path.join(settings.DATA_HOME['mnist'], 'samples', 'Client-0_samples.npz'))
    ld = data['local_data']
    gt = data['ground_truth']
    st = data['stratified_data']
    sy = data['systematic_data']
