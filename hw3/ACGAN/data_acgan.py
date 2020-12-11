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
import csv

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

class DATA(Dataset):
    def __init__(self, args):

        ''' set up basic parameters for dataset '''
        self.data_dir = args.data_dir
    
        ''' set up image trainsform '''
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(MEAN, STD)
        ])
        self.img_list = []

        with open(self.data_dir + "train.csv", "r") as f:
            reader = csv.reader(f, delimiter="\t")
            for i, line in enumerate(reader):
                if i != 0 :
                    line = line[0].split(',')
                    self.img_list.append([self.data_dir + 'train/' +line[0], line[10]])
        print(len(self.img_list))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        img = Image.open(self.img_list[idx][0]).convert('RGB')
        img = self.transform(img)
        class_label = np.asarray(self.img_list[idx][1]).astype(np.float32)
        class_label = torch.tensor(class_label).unsqueeze(0)

        return img, class_label
