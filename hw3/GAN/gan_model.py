import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        def batch1d(out_feat):
            layer = nn.BatchNorm1d(out_feat, 0.8)
            return layer
        def batch2d(out_feat):
            layer = nn.BatchNorm2d(out_feat)
            return layer
        self.hidden1 = nn.Linear(100, 128 * 8 * 8)
        self.batch1 = batch1d(128*8*8)
        self.conv1 = nn.ConvTranspose2d(128, 128, kernel_size = 4, stride = 2, padding = 1)
        self.batch2 = batch2d(128)
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size = 4, stride = 2, padding = 1)
        self.batch3 = batch2d(64)
        self.conv3 = nn.ConvTranspose2d(64, 3, kernel_size = 4, stride = 2, padding = 1)
        self.batch4 = batch2d(3)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = F.relu(self.batch1(self.hidden1(x)))
        x = x.view(-1,128,8,8)  
        x = F.relu(self.batch2(self.conv1(x))) 
        x = F.relu(self.batch3(self.conv2(x))) 
        x = self.tanh(self.batch4(self.conv3(x))) 
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 4, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size = 4, stride = 2, padding = 1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size = 4, stride = 2, padding = 1)
        self.hidden1 =  nn.Linear(4 * 4 * 256, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x)) 
        x = F.relu(self.conv2(x)) 
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x)) 
        x = x.view(x.shape[0], -1)
        x = self.hidden1(x)
        x = self.sig(x)
        return x