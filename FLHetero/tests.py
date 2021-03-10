import json
from time import time

import numpy as np
from FLHetero import temporal_segment, get_segmented_history
from fedlearn import predict_mnist
from pca import get_cluster_list, CPCA
from pca.cluster import clustering_history
from utils import load_history, load_samples, load_weights, load_outputs
from FLHeteroBackend import settings
from matplotlib import pyplot as plt
import os

test_data_file = os.path.join(settings.DATA_DIR, 'test_data.json')
history_file = os.path.join(settings.DATA_DIR, 'history_r50.json')


def test_initialize():
    history = load_history(history_file)
    fed_weight = history['federated']['weight']
    segmentation = temporal_segment(fed_weight, segment_number=20, max_length=5)
    segmented_history = get_segmented_history(history, segmentation)
    data = {'time': segmentation,
            'federated': segmented_history['federated'],
            'others': segmented_history['others'],
            }
    with open(test_data_file, 'w') as f:
        json.dump(data, f)


def test_val():
    # val_file = os.path.join(settings.HISTORY_DIR['mnist'], 'validation.npz')
    val_file = os.path.join(settings.HISTORY_DIR['mnist_mlp'], 'validation.npz')
    data = np.load(val_file)

    client_names = data['client_names']
    loss = data['loss']
    val_acc = data['val_acc']

    n_clients, rounds = loss.shape
    x = np.arange(0, rounds, 1)

    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plt.title('Validate Accuracy')
    for i in range(n_clients):
        plt.plot(x, val_acc[i], label=i)
        print('Client', i, val_acc[i][-1])
    plt.legend()

    plt.subplot(122)
    plt.title('Train Loss')
    for i in range(n_clients):
        plt.plot(x, loss[i], label=i)
    plt.legend()
    plt.show()

    return

    cos_file = os.path.join(settings.HISTORY_DIR['mnist'], 'weights', 'cosines.npz')
    cosines = np.load(cos_file)

    for idx, c_name in enumerate(cosines):
        plt.plot(x, cosines[c_name], label=idx)
    plt.legend()
    plt.show()

    y = np.zeros(200, dtype=np.float32)
    y[0] = 1
    for i in range(1, 200):
        y[i] = y[i - 1] * 0.995
    plt.plot(x, y)
    plt.show()

    # for i in range(n_clients):
    #     plt.plot(x, val_acc[i], label=i)
    # plt.legend()
    # plt.show()


def test_weights(dataset_name, client_name, n_rounds):
    weight = load_weights(dataset_name=dataset_name, client_name=client_name, n_rounds=n_rounds)
    weight_0, weights_server, weights_client = weight['weights_0'], weight['weights_server'], weight['weights_client']
    weight_0, weights_server, weights_client, idx = clustering_history(weight_0=weight_0, weights_server=weights_server,
                                                                       weights_client=weights_client)
    data = {'weight0': weight_0.tolist(),
            'weightsServer': weights_server.tolist(),
            'weightsClient': weights_client.tolist(),
            'splitPoints': idx.tolist(),
            }
    for (k, v) in data.items():
        print('{}: {} {}'.format(k, type(v), np.shape(v)))
    print(data['splitPoints'])
    with open(os.path.join(settings.DATA_DIR, 'test_weights.json'), 'w') as f:
        json.dump(data, f)


def test_cpca(dataset_name, client_name, cm_round, cluster_id, alpha=None):
    data, ground_truth = load_samples(datasets=dataset_name, client_name=client_name, sampling_type='systematic')
    output_labels = load_outputs(datasets=dataset_name, client_name=client_name, cm_round=cm_round)

    outputs_server = output_labels['outputs_server']
    outputs_client = output_labels['outputs_client']
    cluster_list = get_cluster_list(n_clusters=20, data=data,
                                    ground_truth=ground_truth,
                                    outputs_server=outputs_server,
                                    outputs_client=outputs_client)
    hetero_idx = cluster_list[cluster_id]['heteroIndex']
    homo_idx = outputs_server == outputs_client
    bg = data[hetero_idx]
    fg = np.concatenate((data[homo_idx], bg))

    print('cPCA start')
    cPCA = CPCA(n_components=2)
    last_time = time()

    cPCA.fit(target=fg, background=bg)
    print('cPCA fitted in {}s'.format(time() - last_time))
    last_time = time()

    for i in range(4):
        cPCA.transform(i)
    print('cPCA transformed in {}s'.format(time() - last_time))

    data = {'alpha': cPCA.alpha,
            'cpc1': cPCA.components_[0].tolist(),
            'cpc2': cPCA.components_[1].tolist()}
    for (k, v) in data.items():
        print('{}: {} {}'.format(k, type(v), np.shape(v)))

    json_data = json.dumps(data)
    print(json_data)


def test_client(dataset, client_idx):
    history = load_history(dataset)
    loss = history['loss'][client_idx]
    val_acc = history['val_acc'][client_idx]

    data = {
        'loss': loss.tolist(),
        'valAcc': val_acc.tolist(),
    }

    for (k, v) in data.items():
        print('{}: {} {}'.format(k, type(v), np.shape(v)))

    json_data = json.dumps(data)
    print(json_data)


if __name__ == '__main__':
    # test_identify(client_name='Client-0', cm_round=19, n_cluster=20, sampling_type='systematic')
    # check_identify_data()
    # show_loss_and_val_acc('Client-0', 500)
    # test_cpca(dataset_name='mnist', client_name='Client-0', cm_round=199, cluster_id=0)
    # test_client('mnist', 0)
    # test_auto_clustering(dataset_name='mnist', client_name='Client-3', cm_round=199, )
    test_val()
