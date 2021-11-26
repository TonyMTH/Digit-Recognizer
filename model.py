from torch import nn
import torch.nn.functional as F
from collections import OrderedDict


def model(input_size, hidden_sizes, output_size):
    return nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_sizes[0])),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(hidden_sizes[1], hidden_sizes[2])),
        ('relu3', nn.ReLU()),
        ('output', nn.Linear(hidden_sizes[2], output_size)),
        ('softmax', nn.Softmax(dim=1))
    ]))


class Net(nn.Module):
    def __init__(self, output_size):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, 5)
        self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(32, 64, 5)

        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        # in_size = x.shape[0]
        print(x.unsqueeze(1).shape)
        x = self.pool(F.relu(self.conv1(x)))  # N1=28,F=5,P=0,S=1 hence N2=(28-5+2*0)/1 + 1 = 24 24/2 12
        x = self.pool(F.relu(self.conv2(x)))  # N1=12,F=5,P=0,S=1 hence N2=(12-5+2*0)/1 + 1 = 8 8/2 4
        # 64*4*4
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x
