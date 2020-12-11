import os
import torch
import numpy as np
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys
sys.path.insert(1, 'p2/')

import parser
import data
import data_feat_hw

from P2 import P2

import shutil

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def feat_extract(model, loader, device, phase, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for batch, x in enumerate(loader):  # batch_size = 1
        x = x.to(device)
        feat = model.forward_feat(x)
        feat = feat.reshape(x.shape[0], x.shape[1], -1)

        feat_arr = feat[0].cpu().detach().numpy()
        np.savetxt(os.path.join(save_dir, 'feat_' + str(phase) + '_' + str(batch) +'.csv'), feat_arr, delimiter=",")

        print( " [Batch %d/%d]  " % ( batch+1, len(loader)),  end = '\r' )


def eval(model, loader, device, save_dir):
    pred_list = []
    f = open(save_dir, 'w')
    for batch, (x, length) in enumerate(loader):
        x = x.to(device)

        pred = model.forward_classify(x.float(), length)
        _, pred = torch.max(pred, dim = 1)
        for i in range(pred.shape[0]):
            # pred_list.append(int(pred[i].item()))
            f.write(str(int(pred[i].item())))
            f.write('\n')

    # pred_arr = np.asarray(pred_list)
    # np.savetxt(save_dir, pred_arr, delimiter="\n")


if __name__=='__main__':
    args = parser.arg_parse()
    P2 = P2(args, device)
    P2.load_hw_model('p2_cl.pth', 'p2_rnn.pth', device)
    P2.rnn.eval()
    P2.label_cl.eval()

    print('===> prepare dataloader ...')
    valid_loader = torch.utils.data.DataLoader(
        data.DATA(args.hw_video_dir, args.hw_label_dir), 
        batch_size=1, num_workers=args.workers, shuffle=False
    )    
    print('===> extracting features ...')    
    feat_extract(P2, valid_loader, device, 'valid', 'csv_hw_p2' )

    val_loader = torch.utils.data.DataLoader(
        data_feat_hw.DATA('csv_hw_p2', args.video_len, 'valid'), 
        batch_size=args.batch_size, num_workers=args.workers, shuffle=False
    )

    eval(P2, val_loader, device, os.path.join(args.hw_out_dir, 'p2_result.txt'))
    shutil.rmtree('csv_hw_p2' , ignore_errors=True)

