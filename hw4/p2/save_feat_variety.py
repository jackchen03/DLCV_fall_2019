import os
import torch
import numpy as np

import parser
import data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from P2 import P2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def feat_extract(model, loader, device, phase, epoch):
    save_dir = 'csv_' + str(epoch) + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    feat_list = []

    for batch, x in enumerate(loader):  # batch_size = 1
        x = x.to(device)
        feat = model.forward_feat(x)
        feat = feat.reshape(x.shape[0], x.shape[1], -1)

        feat_arr = feat[0].cpu().detach().numpy()
        np.savetxt(os.path.join(save_dir, 'feat_' + str(phase) + '_' + str(batch) +'.csv'), feat_arr, delimiter=",")

        print( " [Batch %d/%d]  " % ( batch+1, len(loader)),  end = '\r' )

if __name__=='__main__':
    args = parser.arg_parse()
    P2 = P2(args, device)

    for epoch in range(2,30):
        print("epoch " + str(epoch+1) + " starts.")
        print('===> prepare dataloader ...')
        train_loader = torch.utils.data.DataLoader(
            data.DATA(os.path.join(args.data_dir, 'train'), os.path.join(args.label_dir, 'gt_train.csv')), 
            batch_size=1, num_workers=args.workers, shuffle=False
        )
        # valid_loader = torch.utils.data.DataLoader(
        #     data.DATA(os.path.join(args.data_dir, 'valid'), os.path.join(args.label_dir, 'gt_valid.csv') ), 
        #     batch_size=1, num_workers=args.workers, shuffle=False
        # )
        feat_extract(P2, train_loader, device, 'train', epoch )
        # feat_extract(P2, valid_loader, device, 'valid', epoch )
    

    

