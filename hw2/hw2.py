import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from baseline_model import Net

import mean_iou_evaluate

from PIL import Image
import numpy as np


import argparse
import cv2

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def arg_parse():
    parser = argparse.ArgumentParser(description='image segmentation for dlcv hw2')

    # Datasets parameters
    parser.add_argument('--data_dir', type=str, default='data/test/img/', 
                    help="root path to data directory")
    parser.add_argument('--output_dir', type = str, default = 'hw2_baseline_output/')

    # baseline model settings
    parser.add_argument('--norm_fn', type=str, default='none', 
                    help="norm layer for trans convolution")
    parser.add_argument('--acti_fn', type=str, default='relu', 
                    help="activation layer for trans convolution")
    
    # resume trained model
    parser.add_argument('--resume', type=str, default='model_baseline.pth.tar', 
                    help="path to the trained model")

    args = parser.parse_args()

    return args

class DATA_HW2(Dataset):
    def __init__(self, args):

        ''' set up basic parameters for dataset '''
        self.data_dir = args.data_dir
        self.image_list = os.listdir(self.data_dir)

        self.transform = transforms.Compose([
                               transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                               transforms.Normalize(MEAN, STD)
                               ])

    def __len__(self):

        return len(self.image_list)

    def __getitem__(self, idx):

        ''' read image '''
        img = Image.open(os.path.join(self.data_dir, self.image_list[idx])).convert('RGB')
        img = self.transform(img)

        return img, self.image_list[idx]

def predict(args, model, data_loader):

    ''' set model to evaluate mode '''
    model.eval()
    with torch.no_grad():
        for idx, (imgs, img_name) in enumerate(data_loader):
            imgs = imgs.to(device)
            pred = model(imgs)

            _, pred = torch.max(pred, dim = 1)
            pred = pred.cpu().numpy().squeeze()

            for i in range(imgs.shape[0]):
                cv2.imwrite(os.path.join(args.output_dir, img_name[i]), pred[i])

if __name__ == '__main__':
    
    args = arg_parse()

    ''' setup GPU '''
    model = Net(args).to(device)

    ''' prepare data_loader '''
    print('===> prepare data loader ...')
    test_loader = torch.utils.data.DataLoader(DATA_HW2(args),
                                              batch_size=32, 
                                              num_workers=4,
                                              shuffle=True)

    ''' resume save model '''
    checkpoint = torch.load(args.resume, map_location = device)
    model.load_state_dict(checkpoint)

    predict(args, model, test_loader)
