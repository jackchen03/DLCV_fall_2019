import os
import json
import torch
import scipy.misc

import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image

import numpy as np

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

class DATA(Dataset):
    def __init__(self, args, mode='train'):

        ''' set up basic parameters for dataset '''
        self.mode = mode
        self.data_dir = args.data_dir
        self.num_classes = 9

        # set up dataset 
        self.img_path = os.path.join(self.data_dir, self.mode, 'img')
        self.label_path = os.path.join(self.data_dir, self.mode, 'seg')

        # flip or not
        self.vflip = args.vflip
        
        ''' set up image trainsform '''
        if self.mode == 'train':
            if args.color_jitter :
                self.transform = transforms.Compose([
                               # transforms.RandomHorizontalFlip(0.5),
                               transforms.ColorJitter(),
                               transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                               transforms.Normalize(MEAN, STD)
                               ])
            else: 
                self.transform = transforms.Compose([
                               # transforms.RandomHorizontalFlip(0.5),
                               transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                               transforms.Normalize(MEAN, STD)
                               ])
            self.transform_l = transforms.Compose([
                                  transforms.ToTensor()
                               ])

        elif self.mode == 'val' or self.mode == 'test':
            self.transform = transforms.Compose([
                               transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                               transforms.Normalize(MEAN, STD)
                               ])

    def __len__(self):
        if self.mode == 'train': number = 5460
        elif self.mode == 'val': number = 500
        elif self.mode == 'test': number = 500

        return number

    def __getitem__(self, idx):
        if (idx > 999):   number = str(idx)
        elif (idx > 99):  number = str(0) + str(idx)
        elif (idx > 9):   number = str(0) + str(0) + str(idx)
        else:             number = str(0) + str(0) + str(0) + str(idx)
        img_dir = os.path.join(self.img_path, number + '.png')  
        label_dir = os.path.join(self.label_path, number +'.png')
        
        ''' read image '''
        img = Image.open(img_dir).convert('RGB')
        label = Image.open(label_dir)

        # random flip
        if self.mode == 'train':
            if self.vflip:
                p_num = np.random.randint(1001)
                if p_num > 500:
                    img = transforms.functional.vflip(img)
                    label = transforms.functional.vflip(label)

        # q_num = np.random.randint(1001)
        # if q_num > 200:
        #     transforms.functional.hflip(img)
        #     transforms.functional.hflip(label)
        img = self.transform(img)
        label = np.asarray(label)   # need RGB??
        label = torch.LongTensor(label)
        # label_one_hot = torch.zeros(self.num_classes, label.shape[1], label.shape[2]).scatter_(0, label, 1)
        # label_one_hot = label_one_hot.long()
        # label_one_hot = label_one_hot.squeeze(0)
        # print(label_one_hot.shape)
        # print(img.shape)

        return img, label
