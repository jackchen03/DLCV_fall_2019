import torch
import torch.nn as nn
import torch.nn.functional as F


class f_ext(nn.Module):
    def __init__(self):
        super(f_ext, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.batch = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(0.4)

    def forward(self, x):
        x = self.maxpool(F.relu(self.batch(self.conv1(x))))
        x = self.maxpool(F.relu(self.batch(self.dropout((self.conv2(x))))))
        x = self.maxpool(F.relu(self.batch(self.dropout((self.conv3(x))))))

        return x


class label_cl(nn.Module):
    def __init__(self):
        super(label_cl, self).__init__()
        self.hidden1 =  nn.Linear(64 * 3 * 3, 256)
        self.hidden2 = nn.Linear(256, 256)
        self.hidden3 = nn.Linear(256, 10)
        # self.batch = nn.BatchNorm1d(100)
        # self.dropout = nn.Dropout2d()

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.hidden3(x)
        return x

class domain_cl(nn.Module):
    def __init__(self):
        super(domain_cl, self).__init__()
        self.hidden1 =  nn.Linear(64*3*3, 256)
        self.hidden2 = nn.Linear(256, 256)
        self.hidden3 = nn.Linear(256, 1)
        # self.batch = nn.BatchNorm1d(100)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.sig(self.hidden3(x))
        return x