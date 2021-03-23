import os

import matplotlib.pyplot as plt
import numpy as np

from FLHeteroBackend.settings import BASE_DIR, DATA_HOME
from cluster import get_cluster_list
from utils import load_samples, load_outputs

output_dir = os.path.join(BASE_DIR, 'data', 'images')


def clear_dir(path):
    for filename in os.listdir(path):
        file = os.path.join(path, filename)
        if os.path.isfile(file):
            os.remove(file)
        else:
            clear_dir(file)
            os.rmdir(file)


def save_images(cluster_id, data, data_idx, data_shape):
    cluster_output_dir = os.path.join(output_dir, 'cluster {} ({})'.format(cluster_id, len(data_idx)))
    if not os.path.exists(cluster_output_dir):
        os.mkdir(cluster_output_dir)

    for image, idx in zip(data, data_idx):
        image_file = os.path.join(cluster_output_dir, '{}.png'.format(idx))
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

    hetero_idx = outputs_server != outputs_client
    cluster_list = get_cluster_list(n_clusters=n_clusters,
                                    dataset=dataset, client_name=client_name, sampling_type=sampling_type,
                                    data=data, outputs_server=outputs_server, outputs_client=outputs_client)

    for cluster_id, cluster_info in enumerate(cluster_list):
        idx = cluster_info['heteroIndex']
        save_images(cluster_id, data[idx], idx, data_shape)


if __name__ == '__main__':
    clear_dir(output_dir)
    export_images_in_clusters(dataset='cifar10',
                              client_name='Client-0',
                              sampling_type='local',
                              cm_round=199,
                              n_clusters=10)
