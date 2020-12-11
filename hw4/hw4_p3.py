import os
import shutil
import torch
import numpy as np
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys
sys.path.insert(1, 'p3/')

import parser
import data_hw
import data_feat_hw

from P3 import P3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def feat_extract(model, loader, device, save_dir_feat):
    if not os.path.exists(save_dir_feat):
        os.makedirs(save_dir_feat)

    for batch, x in enumerate(loader):  # batch_size = 1
        x = x.to(device)
        feat = model.forward_feat(x)

        feat_arr = feat.cpu().detach().numpy()
        np.savetxt(os.path.join(save_dir_feat, 'feat_' + str(batch) +'.csv'), feat_arr, delimiter=",")

        print( " [Batch %d/%d]  " % ( batch+1, len(loader)),  end = '\r' )


def eval(model, valid_dir, device, save_dir):
    

    valid_folders = os.listdir(valid_dir)
    valid_folders.sort()
    for folder in valid_folders:
        pred_list = []
        print(folder)
        valid_loader = torch.utils.data.DataLoader(
            data_feat_hw.DATA(os.path.join(valid_dir, folder)), 
            batch_size=1, num_workers=args.workers, shuffle=False
        )

        f = open(os.path.join(save_dir, folder + '.txt'), 'w')
        for batch, x in enumerate(valid_loader):
            x = x.to(device)

            pred = model.forward_classify(x.float())
            pred = pred.reshape(pred.shape[0]*pred.shape[1], -1)  # (bsize*seq_len, 11)
            _, pred = torch.max(pred, dim = 1)
            for i in range(pred.shape[0]):
                f.write(str(int(pred[i].item())))
                f.write('\n')
                # pred_list.append(int(pred[i].item()))

        # pred_arr = np.asarray(pred_list).astype(np.uint8)
        # save_name = os.path.join(save_dir, folder + '.txt')
        # np.savetxt(save_name, pred_arr, delimiter="\n")

        
        

if __name__=='__main__':
    args = parser.arg_parse()
    P3 = P3(args, device)
    P3.load_hw_model('p3_cl.pth', 'p3_rnn.pth', device)
    P3.rnn.eval()
    P3.label_cl.eval()

    if not os.path.exists('valid_feature_p3'):
        os.makedirs('valid_feature_p3')

    valid_data_dir = args.hw_video_dir
    valid_folders = os.listdir(valid_data_dir)
    valid_folders.sort()
    for folder in valid_folders:
        print('===> prepare dataloader ...')
        valid_loader = torch.utils.data.DataLoader(
            data_hw.DATA(os.path.join(valid_data_dir, folder)), 
            batch_size=256, num_workers=args.workers, shuffle=False
        )
        save_dir_feat = os.path.join('valid_feature_p3', folder)
        feat_extract(P3, valid_loader, device, save_dir_feat)


    save_hw_dir = args.hw_out_dir
    eval(P3, 'valid_feature_p3', device, save_hw_dir)