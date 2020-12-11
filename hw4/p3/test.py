import os
import torch

import parser
import data_feat

from P3 import P3

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def eval(model, valid_dir, device, phase):
    total_num = 0
    acc_num = 0

    valid_folders = os.listdir(valid_dir)
    valid_folders.sort()
    for folder in valid_folders:
        # print('===> prepare dataloader ...')
        feature_dir = os.path.join( phase + '_feature', folder)
        label_dir = os.path.join( phase + '_labels', folder)
        valid_loader = torch.utils.data.DataLoader(
            data_feat.DATA(feature_dir, label_dir), 
            batch_size=1, num_workers=args.workers, shuffle=False
        )

        for batch, (x, label) in enumerate(valid_loader):
            x = x.to(device)
            label = label.to(device)

            pred = model.forward_classify(x.float())
            pred = pred.reshape(pred.shape[0]*pred.shape[1], -1)  # (bsize*seq_len, 11)
            label = label.reshape(-1)       # (bsize*seq_len)
            _, pred = torch.max(pred, dim = 1)
            for i in range(pred.shape[0]):
                total_num += 1
                if pred[i].item() == label[i].item():
                    acc_num += 1

    return acc_num , total_num

 
if __name__ == '__main__':
      
    args = parser.arg_parse()

    P3 = P3(args, device)
    P3.load_model(args.resume_epoch)
    P3.rnn.eval()
    P3.label_cl.eval()

    valid_dir = '../hw4_data/FullLengthVideos/videos/valid/'
    acc_num, total_num = eval(P3, valid_dir, device, 'valid')

    print(str(acc_num) + '/' + str(total_num) + '(' + str(acc_num/total_num) +')' ) 



