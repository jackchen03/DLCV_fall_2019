import os
import torch

import parser
import data_feat

from p1 import P1

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def eval(model, loader, device, total_num):
    acc_num = 0
    for batch, (x, label) in enumerate(loader):
        x = x.to(device)
        label = label.to(device)

        pred = model.forward_classify(x.float())
        _, pred = torch.max(pred, dim = 1)
        for i in range(pred.shape[0]):
            if pred[i].item() == label[i].item():
                acc_num += 1

    return acc_num / total_num

 
if __name__ == '__main__':
      
    args = parser.arg_parse()

    val_loader = torch.utils.data.DataLoader(
        data_feat.DATA('feat_valid.csv', 'label_valid.csv'), 
        batch_size=args.batch_size, num_workers=args.workers, shuffle=False
    )

    P1 = P1(args, device)
    P1.load_model(args.resume_epoch)
    P1.label_cl.eval()

    best_acc = 0
    total_num = 769  # number of validation video


    acc = eval(P1, val_loader, device, total_num)

    print("acc = " + str(acc)) 



