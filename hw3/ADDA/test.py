import os
import torch

import parser
import data

from ADDA import ADDA

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    
    args = parser.arg_parse()

    if args.tran_di == 'sv_mn':
    	loader = torch.utils.data.DataLoader(data.DATA(args, 'test', 'mnistm'),
                                           batch_size=args.batch_size, 
                                           num_workers=args.workers,
                                           shuffle=False)
    if args.tran_di == 'mn_sv':
    	loader = torch.utils.data.DataLoader(data.DATA(args, 'test', 'svhn'),
                                           batch_size=args.batch_size, 
                                           num_workers=args.workers,
                                           shuffle=False)

    ''' model type'''
    model = ADDA(args, device).to(device)
    model.load_model()
    model.target_cnn.eval()
    model.label_cl.eval()

    correct_num = 0
    if args.tran_di == 'sv_mn': leng = 10000
    if args.tran_di == 'mn_sv': leng = 26032

    for batch, (imgs, labels, domains) in enumerate(loader):
        print( " [Batch %d/%d]  " % ( batch+1, len(loader)), end = '\r' )
        imgs = imgs.to(device)
        labels = labels.to(device)
        domains = domains.to(device)

        labels_pred  = model.forward_target(imgs)
        _, labels_pred = torch.max(labels_pred, dim = 1)

        # labels = labels.float()
        for i in range(labels.shape[0]):
            if labels_pred[i] == labels[i]:
                correct_num += 1

    print("acc = " + str(correct_num / leng)) 
    print(correct_num)


