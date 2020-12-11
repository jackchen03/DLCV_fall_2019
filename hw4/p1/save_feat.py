import os
import torch
import numpy as np

import parser
import data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from p1 import P1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def feat_extract(model, loader, device, phase):
    feat_list = []
    label_list = []
    for batch, (x, label) in enumerate(loader):
        x = x.to(device)
        x = x.permute(0,1,3,4,2)

        feat = model.forward_feat(x)

        for i in range(feat.shape[0]):
            feat_list.append(feat[i].cpu().detach().numpy())
            label_list.append(label[i].item())

        print( " [Batch %d/%d]  " % ( batch+1, len(loader)),  end = '\r' )

    feat_arr = np.asarray(feat_list)
    label_arr = np.asarray(label_list).reshape(-1,1)
    np.savetxt('feat_' + str(phase) + '.csv', feat_arr, delimiter=",")
    np.savetxt('label_' + str(phase) + '.csv', label_arr, delimiter=",")

if __name__=='__main__':
    args = parser.arg_parse()
    print('===> prepare dataloader ...')
    train_loader = torch.utils.data.DataLoader(
        data.DATA(os.path.join(args.data_dir, 'train'), os.path.join(args.label_dir, 'gt_train.csv') ), 
        batch_size=256, num_workers=args.workers, shuffle=False
    )
    valid_loader = torch.utils.data.DataLoader(
        data.DATA(os.path.join(args.data_dir, 'valid'), os.path.join(args.label_dir, 'gt_valid.csv') ), 
        batch_size=256, num_workers=args.workers, shuffle=False
    )

    P1 = P1(args, device)

    feat_extract(P1, train_loader, device, 'train')
    feat_extract(P1, valid_loader, device, 'valid')

