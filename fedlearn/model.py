import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from fedlearn.train import train, test
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(31, 20)
        self.fc2 = nn.Linear(20, 8)
        self.fc3 = nn.Linear(8, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Server:
    def __init__(self, test_loader, start_round=1, checkpoint_path='./', device='cpu'):
        self.model = Net().to(device)
        self.device = device
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss()
        self.history = {'weight': [], 'loss': []}
        self.cur_round = start_round
        # self.save_mode = save_mode
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

        # if self.save_mode == 'all':
        #     self.save(os.path.join(self.checkpoint_path, 'Server_r{}.cp'.format(self.cur_round)))

        weight = None
        for name, paras in self.model.named_parameters():
            p = paras.data.cpu().numpy().reshape(-1)
            weight = p if weight is None else np.concatenate((weight, p))

        self.history['weight'].append(weight.tolist())

        acc, loss = test(test_loader=self.test_loader, model=self.model, criterion=self.criterion, device=self.device)
        self.history['loss'].append(loss)
        print('Fed Acc: {} Fed Loss: {}'.format(acc, loss))

    def save(self, suffix):
        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_path, 'Server' + suffix + '.cp'))

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def summary(self):
        print('Server Summary')


class Client:
    def __init__(self, client_name, train_loader, test_loader, start_epoch=1, checkpoint_path='./',
                 device='cpu'):
        self.client_name = client_name
        self.device = device
        self.model = Net().to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.Adam(self.model.parameters())
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[20, 50], gamma=0.1,
                                                              last_epoch=-1)
        self.history = {'clientName': client_name, 'weight': [], 'loss': []}
        self.cur_epoch = start_epoch
        # self.save_mode = save_mode
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

    def run(self, epochs):
        running_loss = 0.0
        for epoch in range(epochs):
            running_loss = train(train_loader=self.train_loader, model=self.model, criterion=self.criterion,
                                 optimizer=self.optimizer, scheduler=self.scheduler, device=self.device)

            print('Epoch {}: loss - {}'.format(self.cur_epoch, running_loss))
            self.cur_epoch += 1

        weight = None
        for name, paras in self.model.named_parameters():
            p = paras.data.cpu().numpy().reshape(-1)
            weight = p if weight is None else np.concatenate((weight, p))
        self.history['weight'].append(weight.tolist())
        self.history['loss'].append(running_loss)

        acc, val_loss = test(test_loader=self.test_loader, model=self.model, criterion=self.criterion,
                             device=self.device)
        print('Accuracy: {} Validate Loss: {}'.format(acc, val_loss))

    def summary(self):
        print('{} Summary'.format(self.client_name))
        # print(self.history)
