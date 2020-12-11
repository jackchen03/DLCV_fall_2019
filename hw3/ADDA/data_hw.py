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
    def __init__(self, data_dir):

        ''' set up basic parameters for dataset '''
        self.data_dir = data_dir

        ''' set up image trainsform '''
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(MEAN, STD)
        ])

        self.img_list = []
        lst = os.listdir(self.data_dir)
        lst.sort()
        for file in lst:
            self.img_list.append(file) 

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        img = Image.open(os.path.join(self.data_dir, self.img_list[idx])).convert('L')
        img = self.transform(img)
        img_name = self.img_list[idx]


        return img, img_name
