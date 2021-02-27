import os

import numpy as np
import json

from FLHeteroBackend import settings


def load_samples0(filename):
    data = np.load(filename)
    outputs_server = data['outputs_server']
    outputs_client = data['outputs_client']
    ground_truth = data['ground_truth']
    local_data = data['local_data']
    samples = data['samples']
    return samples, local_data, ground_truth, outputs_client, outputs_server


def load_samples(datasets, client_name, sampling_type):
    samples_file = os.path.join(settings.DATA_HOME[datasets], 'samples', '{}_samples.npz'.format(client_name))
    samples_data = np.load(samples_file)
    samples = {
        'local': samples_data['local_data'],
        'stratified': samples_data['stratified_data'],
        'systematic': samples_data['systematic_data'],
    }
    ground_truth = samples_data['ground_truth']

    return samples[sampling_type], ground_truth


def load_outputs(datasets, client_name, cm_round):
    outputs_file = os.path.join(settings.HISTORY_DIR[datasets], 'outputs',
                                '{}_outputs_server_r{}.npy'.format(client_name, cm_round))
    outputs_server = np.load(outputs_file)
    outputs_file = os.path.join(settings.HISTORY_DIR[datasets], 'outputs',
                                '{}_outputs_client.npy'.format(client_name))
    outputs_client = np.load(outputs_file)
    return {'outputs_server': outputs_server, 'outputs_client': outputs_client}


def chebyshev_distance(x, y):
    return np.max(np.abs(x - y))


def euclidean_distance(x, y):
    return np.linalg.norm(x - y)


def grid_by_percentile(x, n=10, return_quantiles=False):
    step = 100 // n
    percentiles = np.arange(0, 100, step) + step
    quantiles = [np.percentile(x, q, interpolation='lower') for q in percentiles]
    quantiles = np.unique(quantiles)
    y = [np.searchsorted(quantiles, val, 'left') for val in x]
    if return_quantiles:
        return y, quantiles
    return y


def grid_by_value(x, n=10):
    maxi = np.max(x)
    mini = np.min(x)
    x = (x - mini) / (maxi - mini) * n
    x = np.floor(x).astype(int)
    return x


def load_weights(dataset_name, client_name, n_rounds):
    data_home = os.path.join(settings.HISTORY_DIR[dataset_name], 'weights')
    weights_0 = np.load(os.path.join(data_home, 'weights_0.npy'))
    weights_client = np.array([np.load(os.path.join(data_home, '{}_weights_r{}.npy'.format(client_name, r))) for r in
                               range(n_rounds)])
    weights_server = np.array(
        [np.load(os.path.join(data_home, 'Server_weights_r{}.npy'.format(r))) for r in range(n_rounds)])
    return {'weights_0': weights_0, 'weights_client': weights_client, 'weights_server': weights_server}


def load_history(dataset_name):
    data = np.load(os.path.join(settings.HISTORY_DIR[dataset_name], 'validation.npz'))
    client_names = data['client_names']
    n_clients = client_names.shape[0]
    loss = data['loss']
    val_acc = data['val_acc']
    n_rounds = loss.shape[0]
    return {'client_names': client_names,
            'n_clients': n_clients,
            'loss': loss,
            'val_acc': val_acc,
            'n_rounds': n_rounds,
            }
