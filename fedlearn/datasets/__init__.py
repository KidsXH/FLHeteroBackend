import torch
from torch.utils.data.dataset import T_co
from torch.utils.data import DataLoader
import numpy as np
import os


class AmericanData(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        super(AmericanData, self).__init__()
        self.data = data
        self.labels = labels

    def __getitem__(self, index) -> T_co:
        feat, label = self.data[index], self.labels[index]
        return feat, label

    def __len__(self):
        return self.data.shape[0]


def feature_normalize(data):
    mu = np.mean(data)
    std = np.std(data)
    return (data - mu) / std


def load_data(path):
    train = np.genfromtxt(path + '_train.csv', delimiter=',', skip_header=1)
    test = np.genfromtxt(path + '_test.csv', delimiter=',', skip_header=1)
    train_data = feature_normalize(train[:, 1:])
    train_labels = train[:, 0] - 1
    test_data = feature_normalize(test[:, 1:])
    test_labels = test[:, 0] - 1
    return train_data, train_labels, test_data, test_labels


def load_all_test(path, regions):
    data = None
    labels = None
    for region_name in regions:
        filename = os.path.join(path, 'AmericanData', region_name) + '_test.csv'
        test = np.genfromtxt(filename, delimiter=',', skip_header=1)
        test_data = feature_normalize(test[:, 1:])
        test_labels = test[:, 0] - 1
        data = test_data if data is None else np.concatenate((data, test_data))
        labels = test_labels if labels is None else np.concatenate((labels, test_labels))
    return data, labels


def load_region(path):
    train = np.genfromtxt(path + '_train.csv', delimiter=',', skip_header=1)
    test = np.genfromtxt(path + '_test.csv', delimiter=',', skip_header=1)
    train_data = feature_normalize(train[:, 1:])
    train_labels = train[:, 0] - 1
    test_data = feature_normalize(test[:, 1:])
    test_labels = test[:, 0] - 1
    return np.concatenate((train_data, test_data)), np.concatenate((train_labels, test_labels))


def get_american_data_loader(path, region_name, batch_size):
    train_data, train_labels, test_data, test_labels = load_data(os.path.join(path, 'AmericanData', region_name))
    train_dataset = AmericanData(data=train_data, labels=train_labels)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataset = AmericanData(data=test_data, labels=test_labels)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    print('AmericanData - ' + region_name + ' loaded.')

    return train_loader, test_loader


def get_all_test_data_loader(path, regions, batch_size):
    test_data, test_labels = load_all_test(path, regions)
    test_dataset = AmericanData(data=test_data, labels=test_labels)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return test_loader


def get_region_data_loader(path, region, batch_size):
    data, labels = load_region(os.path.join(path, 'AmericanData', region))
    dataset = AmericanData(data=data, labels=labels)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return data_loader
