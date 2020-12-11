import os
import torch

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import cv2

import sys
sys.path.insert(1, 'DANN/')

import parser
import data_hw
from DANN import DANN





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    
    args = parser.arg_parse()

    loader = torch.utils.data.DataLoader(data_hw.DATA(args.hw_data_dir),
                                           batch_size=args.batch_size, 
                                           num_workers=args.workers,
                                           shuffle=False)

    if args.tar_domain =='mnistm':
        f_path = 'DANN_svmn_F.pth'
        l_path = 'DANN_svmn_L.pth'
    if args.tar_domain =='svhn':
        f_path = 'DANN_mnsv_F.pth'
        l_path = 'DANN_mnsv_L.pth'
    ''' model type'''
    model = DANN(args).to(device)
    model.load_test_model(f_path, l_path, device)

    img_name_list = []
    label_list = []

    for batch, (img, img_name) in enumerate(loader):
        img = img.to(device)

        labels_pred, _  = model.forward(img, 0)
        _, labels_pred = torch.max(labels_pred, dim = 1)

        for i in range(labels_pred.shape[0]):
            img_name_list.append(img_name[i])
            label_list.append(labels_pred[i].item())

    data = {'image_name': img_name_list, 'label': label_list}
    df = pd.DataFrame.from_dict(data)
    df.to_csv(args.out_path, index = False)



