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
sys.path.insert(1, 'ADDA/')

import parser
import data_hw
from ADDA import ADDA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    
    args = parser.arg_parse()

    loader = torch.utils.data.DataLoader(data_hw.DATA(args.hw_data_dir),
                                           batch_size=args.batch_size, 
                                           num_workers=args.workers,
                                           shuffle=False)
    # if args.tar_domain =='mnistm':
    #     f_path = 'Best_TCNN.pth'
    #     l_path = 'Best_L_cl.pth'
    
    if args.tar_domain =='mnistm':
        f_path = 'ADDA_svmn_TCNN.pth'
        l_path = 'ADDA_svmn_L.pth'
    if args.tar_domain =='svhn':
        f_path = 'ADDA_mnsv_TCNN.pth'
        l_path = 'ADDA_mnsv_L.pth'
    ''' model type'''
    model = ADDA(args, device).to(device)
    model.load_test_model(f_path, l_path, device)
    model.target_cnn.eval()
    model.label_cl.eval()


    img_name_list = []
    label_list = []

    for batch, (img, img_name) in enumerate(loader):
        img = img.to(device)

        labels_pred  = model.forward_target(img)
        _, labels_pred = torch.max(labels_pred, dim = 1)

        for i in range(labels_pred.shape[0]):
            img_name_list.append(img_name[i])
            label_list.append(labels_pred[i].item())

    data = {'image_name': img_name_list, 'label': label_list}
    df = pd.DataFrame.from_dict(data)
    df.to_csv(args.out_path, index = False)



