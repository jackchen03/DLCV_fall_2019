import os
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys
sys.path.insert(1, 'p1/')

import parser
import data
import data_feat

from p1 import P1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def feat_extract(model, loader, device):
    feat_list = []
    label_list = []
    for batch, (x, label) in enumerate(loader):
        x = x.to(device)
        x = x.permute(0,1,3,4,2)

        feat = model.forward_feat(x)

        for i in range(feat.shape[0]):
            feat_list.append(feat[i].cpu().detach().numpy())
            label_list.append(int(label[i].item()))

        print( " [Batch %d/%d]  " % ( batch+1, len(loader)),  end = '\r' )

    feat_arr = np.asarray(feat_list)
    label_arr = np.asarray(label_list).reshape(-1,1)
    np.savetxt('feat_hw_p1.csv', feat_arr, delimiter=",")
    np.savetxt('label_hw_p1.csv', label_arr, delimiter=",")


def eval(model, loader, device, save_dir):
    pred_list = []
    f = open(save_dir, 'w')
    for batch, (x, label) in enumerate(loader):
        x = x.to(device)
        label = label.to(device)

        pred = model.forward_classify(x.float())
        _, pred = torch.max(pred, dim = 1)
        for i in range(pred.shape[0]):
            # pred_list.append(int(pred[i].item()))
            f.write(str(int(pred[i].item())))
            f.write('\n')

    # pred_arr = np.asarray(pred_list)
    # np.savetxt(save_dir, pred_arr, delimiter="\n")

if __name__=='__main__':
    args = parser.arg_parse()
    P1 = P1(args, device)
    P1.load_hw_model('p1_cl.pth', device)
    P1.label_cl.eval()

    print('===> prepare dataloader ...')
    valid_loader = torch.utils.data.DataLoader(
        data.DATA(args.hw_video_dir, args.hw_label_dir), 
        batch_size=256, num_workers=args.workers, shuffle=False
    )
    print('===> extracting features ...')    
    feat_extract(P1, valid_loader, device)

    val_loader = torch.utils.data.DataLoader(
        data_feat.DATA('feat_hw_p1.csv', 'label_hw_p1.csv'), 
        batch_size=16, num_workers=args.workers, shuffle=False
    )

    eval(P1, val_loader, device, os.path.join(args.hw_out_dir, 'p1_valid.txt'))

