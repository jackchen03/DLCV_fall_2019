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

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

class DATA(Dataset):
    def __init__(self, args):

        ''' set up basic parameters for dataset '''
        self.data_dir = args.data_dir

        # set up dataset 
        self.img_path = args.data_dir        
        ''' set up image trainsform '''
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(MEAN, STD)
        ])
        self.img_list = []

        for img in os.listdir(self.img_path):
          self.img_list.append(os.path.join(self.img_path, img))
        print(len(self.img_list))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        img = Image.open(self.img_list[idx]).convert('RGB')
        img = self.transform(img)

        return img
