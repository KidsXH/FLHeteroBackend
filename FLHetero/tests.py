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


def test_identify(client_name, cm_round, n_cluster, sampling_type):
    last_time = time()

    # predict_mnist(client_name, cm_round)

    print('Prediction finished in {}s.'.format(time() - last_time))
    last_time = time()

    samples, ground_truth, outputs_server, outputs_client = load_samples(datasets='mnist', client_name=client_name,
                                                                         sampling_type=sampling_type, cm_round=cm_round)
    print('Loading samples finished in {}s.'.format(time() - last_time))
    last_time = time()

    hetero_list, pca = get_pca_result(n_clusters=n_cluster,
                                      data=samples,
                                      ground_truth=ground_truth,
                                      outputs_client=outputs_client,
                                      outputs_server=outputs_server,
                                      show_cpca_figures=5)

    print('Getting pca result finished in {}s.'.format(time() - last_time))
    last_time = time()

    data = {'groundTruth': ground_truth.tolist(),
            'outputLabels': outputs_server.tolist(),
            'heteroLabels': (outputs_client != outputs_server).tolist(),
            'fedResult': (outputs_server == ground_truth).tolist(),
            'pca': pca,
            'heteroList': hetero_list,
            }
    with open(test_data_file, 'w') as f:
        json.dump(data, f)

    print('Saving finished in {}s.'.format(time() - last_time))


def check_identify_data():
    with open(test_data_file, 'r') as f:
        data = json.load(f)
    local_data = data['localData']
    samples = data['samples']
    ground_truth = data['groundTruth']
    output_labels = data['outputLabels']
    het_labels = data['heteroLabels']
    fed_res = data['fedResult']
    pca = data['pca']
    het_list = data['heteroList']
    print(np.shape(local_data))
    print(np.shape(samples))
    print(np.shape(ground_truth))
    print(np.shape(output_labels))
    print(np.shape(het_labels))
    print(np.shape(fed_res))
    print(pca.keys())
    for value in pca.values():
        print(np.shape(value))

    print(np.shape(het_list))
    print(het_list[0].keys())
    print(het_list[0]['cpca'].keys())
    for value in het_list[0]['cpca'].values():
        print(np.shape(value))
    print(het_list[0]['heteroSize'])
    print(np.shape(het_list[0]['heteroIndex']))
    print(np.unique(het_list[0]['heteroIndex']).shape)
    print(het_list[0]['heteroRate'])


def show_loss_and_val_acc(client_name, total_round):
    with open(os.path.join(settings.DATA_DIR, 'history_{}.json'.format(client_name))) as f:
        data = json.load(f)

    loss = data['Server']['loss']
    acc = data['Server']['val_acc']
    x = np.arange(0, total_round, 1)

    from matplotlib import pyplot as plt
    plt.subplot(221)
    plt.plot(x, loss)
    plt.title('Server Train Loss')
    plt.subplot(222)
    plt.plot(x, acc)
    plt.title('Server Validate Accuracy')

    loss = data['Client']['loss']
    acc = data['Client']['val_acc']
    plt.subplot(223)
    plt.plot(x, loss)
    plt.title('Client Train Loss')
    plt.subplot(224)
    plt.plot(x, acc)
    plt.title('Client Validate Accuracy')
    plt.show()


def test_val():
    val_file = os.path.join(settings.HISTORY_DIR['mnist'], 'validation.npz')
    data = np.load(val_file)

    client_names = data['client_names']
    loss = data['loss']
    val_acc = data['val_acc']

    n_clients, rounds = loss.shape
    x = np.arange(0, rounds, 1)
    for i in range(n_clients):
        plt.plot(x, val_acc[i], label=i)
    plt.legend()
    plt.show()

    for i in range(n_clients):
        plt.plot(x, loss[i])
    plt.show()


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


def test_cpca(dataset_name, client_name, cm_round, cluster_id, alpha=30):
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
    cPCA = CPCA(n_components=2)
    cPCA.fit_transform(target=fg, background=bg, alpha=alpha)

    data = {'cpc1': cPCA.components_[0].tolist(),
            'cpc2': cPCA.components_[1].tolist()}
    for (k, v) in data.items():
        print('{}: {} {}'.format(k, type(v), np.shape(v)))

    json_data = json.dumps(data)


if __name__ == '__main__':
    # test_identify(client_name='Client-0', cm_round=19, n_cluster=20, sampling_type='systematic')
    # check_identify_data()
    # show_loss_and_val_acc('Client-0', 500)
    test_cpca('mnist', 'Client-0', 199, 0)
