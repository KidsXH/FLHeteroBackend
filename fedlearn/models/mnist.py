import torch
from torch import nn

from fedlearn.models.csBase import ClientBase, ServerBase


class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.fc1 = nn.Linear(784, 10)

    def forward(self, x):
        x = self.fc1(x)
        return x


class Client(ClientBase):
    def __init__(self, client_name, train_loader, test_loader, start_epoch=0, checkpoint_path='./', device='cpu'):
        super(Client, self).__init__(client_name, train_loader, test_loader,
                                     start_epoch=start_epoch, checkpoint_path=checkpoint_path, device=device)
        self.model = MnistModel().to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.995, last_epoch=start_epoch - 1)


class Server(ServerBase):
    def __init__(self, test_loader, start_round=0, checkpoint_path='./', device='cpu'):
        super(Server, self).__init__(test_loader, start_round=start_round, checkpoint_path=checkpoint_path,
                                     device=device)
        self.model = MnistModel().to(device)
        self.criterion = nn.CrossEntropyLoss()
