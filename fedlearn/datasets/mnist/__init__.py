import json

import torch
from torch.utils.data.dataset import T_co
from torch.utils.data import DataLoader
import numpy as np
import os
from FLHeteroBackend import settings

train_file = os.path.join(settings.DATA_HOME['mnist'], 'train.json')
test_file = os.path.join(settings.DATA_HOME['mnist'], 'test.json')


class MnistData(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        super(MnistData, self).__init__()

        # normalize
        data = np.array(data, dtype=np.float32)
        mu = np.mean(data, 0)
        sigma = np.std(data, 0)
        data = (data - mu) / (sigma + 0.001)

        self.data = data
        self.labels = labels

    def __getitem__(self, index) -> T_co:
        feat, label = self.data[index], self.labels[index]
        return feat, label

    def __len__(self):
        return self.data.shape[0]


def get_mnist_data(batch_size=100):
    """
    Get MNIST data.
    :param batch_size: batch size
    :return: client_names, train_loaders, test_loaders, server_data_loader
    """
    with open(train_file) as f:
        train_data = json.load(f)
    with open(test_file) as f:
        test_data = json.load(f)
    client_names = train_data['username']
    n_clients = len(client_names)
    train_loaders = []
    test_loaders = []
    client_labels = []

    for i in range(n_clients):
        train_dataset = MnistData(data=train_data['data'][i], labels=train_data['target'][i])
        train_loaders.append(DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4))
        test_dataset = MnistData(data=test_data['data'][i], labels=test_data['target'][i])
        test_loaders.append(DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=4))
        client_labels.append(np.unique(train_data['target'][i]))

    print(client_labels)
    server_data = np.concatenate((test_data['data']))
    server_labels = np.concatenate((test_data['target']))
    server_dataset = MnistData(data=server_data, labels=server_labels)
    server_data_loader = DataLoader(dataset=server_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return client_names, train_loaders, test_loaders, server_data_loader


def get_mnist_client_data(client_name, batch_size):
    """
    :param client_name:
    :param batch_size:
    :return:
    """
    with open(train_file) as f:
        train_data = json.load(f)
    with open(test_file) as f:
        test_data = json.load(f)
    client_names = train_data['username']
    n_clients = len(client_names)
    data_loader = None

    for i in range(n_clients):
        if client_name == client_names[i]:
            data = np.concatenate((train_data['data'][i], test_data['data'][i]))
            labels = np.concatenate((train_data['target'][i], test_data['target'][i]))
            dataset = MnistData(data=data, labels=labels)
            data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    if data_loader is None:
        raise ValueError('Invalid client name. Client name must be in {}.'.format(client_names))

    return data_loader
