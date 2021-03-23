import os

import matplotlib.pyplot as plt
import numpy as np

from FLHeteroBackend.settings import BASE_DIR, DATA_HOME
from cluster import get_cluster_list
from utils import load_samples, load_outputs

output_dir = os.path.join(BASE_DIR, 'data', 'images')


def init():
    clear_dir(output_dir)
    os.mkdir(output_dir)
    os.mkdir(os.path.join(output_dir, 'dist'))
    os.mkdir(os.path.join(output_dir, 'rank'))


def clear_dir(path):
    if not os.path.exists(path):
        return
    for filename in os.listdir(path):
        file = os.path.join(path, filename)
        if os.path.isfile(file):
            os.remove(file)
        else:
            clear_dir(file)
    os.rmdir(path)


def save_images(base_dir, data, data_idx, data_shape):
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    for image, idx in zip(data, data_idx):
        image_file = os.path.join(base_dir, '{}.png'.format(idx))
        image = image.reshape(data_shape)
        if len(data_shape) == 3:
            image = image.transpose((1, 2, 0))
            plt.imsave(image_file, image)
        else:
            plt.imsave(image_file, image, cmap='gray')


def export_images_in_clusters(dataset, client_name, sampling_type, cm_round, n_clusters):
    sample_data = np.load(os.path.join(DATA_HOME[dataset], 'samples.npz'), allow_pickle=True)
    client_id = np.where(sample_data['client_names'] == client_name)[0][0]
    data = sample_data[sampling_type][client_id]
    data_shape = sample_data['shape']
    output = load_outputs(dataset, client_name, sampling_type, cm_round)
    outputs_server, outputs_client = output['outputs_server'], output['outputs_client']

    cluster_list_dist = get_cluster_list(n_clusters=n_clusters, method='distance',
                                         dataset=dataset, client_name=client_name, sampling_type=sampling_type,
                                         data=data, outputs_server=outputs_server, outputs_client=outputs_client)

    for cluster_id, cluster_info in enumerate(cluster_list_dist):
        idx = cluster_info['heteroIndex']
        size = cluster_info['heteroSize']
        base_dir = os.path.join(output_dir, 'dist', 'cluster {} ({})'.format(cluster_id, size))
        save_images(base_dir, data[idx], idx, data_shape)

    cluster_list_rank = get_cluster_list(n_clusters=n_clusters, method='rank',
                                         dataset=dataset, client_name=client_name, sampling_type=sampling_type,
                                         data=data, outputs_server=outputs_server, outputs_client=outputs_client)

    for cluster_id, cluster_info in enumerate(cluster_list_rank):
        idx = cluster_info['heteroIndex']
        size = cluster_info['heteroSize']
        base_dir = os.path.join(output_dir, 'rank', 'cluster {} ({})'.format(cluster_id, size))
        save_images(base_dir, data[idx], idx, data_shape)


if __name__ == '__main__':
    init()
    export_images_in_clusters(dataset='mnist',
                              client_name='Client-0',
                              sampling_type='local',
                              cm_round=199,
                              n_clusters=None)
