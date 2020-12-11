import os
import json
import torch
import scipy.misc

import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image
import cv2

import numpy as np
import pandas as pd

MEAN = [0.5]
STD = [0.5]

class DATA(Dataset):
    def __init__(self, args, mode = 'train', domain = 'svhn' ):

        ''' set up basic parameters for dataset '''
        self.data_dir = args.data_dir
        if mode == 'train' or mode == 'val':
            self.mode = 'train'
        if mode == 'test':
            self.mode = 'test'
        self.phase = mode
        self.domain = domain

        ''' set up image trainsform '''
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(MEAN, STD)
        ])
        img_list = []

        df = pd.read_csv(os.path.join(self.data_dir, self.domain+ '/', self.mode + '.csv'))
        labels = df['label']
        img_list = df['image_name']

        if self.mode == 'train':
            if self.phase == 'train':
                if self.domain == 'svhn':
                    self.img_list = img_list[:67257]
                    self.labels = labels[:67257]
                if self.domain == 'mnistm':
                    self.img_list = img_list[:55000]
                    self.labels = labels[:55000]
            if self.phase == 'val':
                if self.domain == 'svhn':
                    self.img_list = img_list[67257:]
                    self.labels = labels[67257:]
                if self.domain == 'mnistm':
                    self.img_list = img_list[55000:]
                    self.labels = labels[55000:]
        if self.mode == 'test':
            self.img_list = img_list
            self.labels = labels

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        if self.phase == 'train' or self.phase == 'test': start = 0
        if self.phase == 'val':
            if self.domain == 'svhn': start = 67257
            if self.domain == 'mnistm': start = 55000

        img = Image.open(os.path.join(self.data_dir, self.domain+ '/', self.mode + '/', self.img_list[idx + start] )).convert('L')
        img = self.transform(img)

        label = torch.tensor(self.labels[idx + start])

        if self.domain == 'svhn':
            domain = torch.zeros(1)
        if self.domain == 'mnistm':
            domain = torch.ones(1)

        return img, label, domain
