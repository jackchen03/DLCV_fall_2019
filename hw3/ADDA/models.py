import torch
import torch.nn as nn
import torch.nn.functional as F


class source_cnn(nn.Module):
    def __init__(self):
        super(source_cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
        self.conv2 = nn.Conv2d(20, 50, kernel_size=(5, 5), stride=(1, 1))
        self.maxpool = nn.MaxPool2d (2)
        self.dropout = nn.Dropout2d(0.5)
        self.linear = nn.Linear(800, 500)

    def forward(self, x):
        x = F.relu(self.maxpool(self.conv1(x)))
        x = F.relu(self.maxpool(self.dropout((self.conv2(x)))))
        x = x.view(x.shape[0], -1)
        x = self.linear(x)

        return x

class target_cnn(nn.Module):
    def __init__(self):
        super(target_cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
        self.conv2 = nn.Conv2d(20, 50, kernel_size=(5, 5), stride=(1, 1))
        self.maxpool = nn.MaxPool2d (2)
        self.dropout = nn.Dropout2d(0.5)
        self.linear = nn.Linear(800, 500)

    def forward(self, x):
        x = F.relu(self.maxpool(self.conv1(x)))
        x = F.relu(self.maxpool(self.dropout((self.conv2(x)))))
        x = x.view(x.shape[0], -1)
        x = self.linear(x)

        return x


class label_cl(nn.Module):
    def __init__(self):
        super(label_cl, self).__init__()
        # self.hidden1 =  nn.Linear(64 * 3 * 3, 256)
        # self.hidden2 = nn.Linear(256, 256)
        self.hidden = nn.Linear(500, 10)
        # self.batch = nn.BatchNorm1d(100)
        self.dropout = nn.Dropout2d()

    def forward(self, x):
        # x = F.relu(self.hidden1(x))
        # x = F.relu(self.hidden2(x))
        x = self.dropout(F.relu(x))
        x = self.hidden(x)
        return x

class domain_cl(nn.Module):
    def __init__(self):
        super(domain_cl, self).__init__()
        self.hidden1 = nn.Linear(500,500)
        self.hidden2 = nn.Linear(500,500)
        self.hidden3 = nn.Linear(500,1)
        # self.hidden1 =  nn.Linear(64*3*3, 256)
        # self.hidden2 = nn.Linear(256, 256)
        # self.hidden3 = nn.Linear(256, 1)
        # self.batch = nn.BatchNorm1d(100)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.sig(self.hidden3(x))
        return x