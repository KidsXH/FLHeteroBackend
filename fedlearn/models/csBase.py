import os

import torch
import numpy as np

from fedlearn.train import train, test


class ClientBase:
    def __init__(self, client_name, train_loader, test_loader, start_epoch,
                 checkpoint_path, device):
        self.client_name = client_name
        self.device = device
        self.model = None
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.history = {'clientName': client_name, 'weight': [], 'loss': []}
        self.cur_epoch = start_epoch
        self.checkpoint_path = checkpoint_path
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)

    def get_client_name(self):
        return self.client_name

    def get_parameters(self):
        return self.model.state_dict()

    def set_parameters(self, parameters):
        self.model.load_state_dict(parameters)

    def save(self, suffix):
        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_path, self.client_name + suffix + '.cp'))

    def run(self, epochs, save_history=True, validate=True):
        running_loss = 0.0
        for epoch in range(epochs):
            running_loss = train(train_loader=self.train_loader, model=self.model, criterion=self.criterion,
                                 optimizer=self.optimizer, scheduler=self.scheduler, device=self.device)

            print('Epoch {}: loss - {}'.format(self.cur_epoch, running_loss))
            self.cur_epoch += 1

        if save_history:
            weight = None
            for name, paras in self.model.named_parameters():
                p = paras.data.cpu().numpy().reshape(-1)
                weight = p if weight is None else np.concatenate((weight, p))
            self.history['weight'].append(weight.tolist())
            self.history['loss'].append(running_loss)

        if validate:
            acc, val_loss = test(test_loader=self.test_loader, model=self.model, criterion=self.criterion,
                                 device=self.device)
            print('Accuracy: {} Validate Loss: {}'.format(acc, val_loss))


class ServerBase:
    def __init__(self, test_loader, start_round, checkpoint_path, device):
        self.model = None
        self.device = device
        self.test_loader = test_loader
        self.criterion = None
        self.history = {'weight': [], 'loss': []}
        self.cur_round = start_round
        self.checkpoint_path = checkpoint_path
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)

    def get_parameters(self):
        return self.model.state_dict()

    def aggregate(self, local_parameters):
        """Aggregate parameters using FedAvg
        """
        clients_number = len(local_parameters)
        aggregated_parameters = local_parameters[0]

        for parameters in local_parameters[1:]:
            for item in parameters:
                aggregated_parameters[item] += parameters[item]

        for item in aggregated_parameters:
            aggregated_parameters[item] /= clients_number

        self.model.load_state_dict(aggregated_parameters)

        acc, loss = test(test_loader=self.test_loader, model=self.model, criterion=self.criterion, device=self.device)
        print('Fed Acc: {} Fed Loss: {}'.format(acc, loss))

        weight = None
        for name, paras in self.model.named_parameters():
            p = paras.data.cpu().numpy().reshape(-1)
            weight = p if weight is None else np.concatenate((weight, p))

        self.history['weight'].append(weight.tolist())
        self.history['loss'].append(loss)

    def save(self, suffix):
        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_path, 'Server' + suffix + '.cp'))

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
