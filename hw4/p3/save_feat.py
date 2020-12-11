import os
import torch
import numpy as np

import parser
import data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from P3 import P3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def feat_extract(model, loader, device, save_dir_feat, save_dir_label):
    if not os.path.exists(save_dir_feat):
        os.makedirs(save_dir_feat)
    if not os.path.exists(save_dir_label):
        os.makedirs(save_dir_label)

    for batch, (x, label) in enumerate(loader):  # batch_size = 1
        x = x.to(device)
        feat = model.forward_feat(x)

        feat_arr = feat.cpu().detach().numpy()
        label_arr = label.detach().numpy()
        np.savetxt(os.path.join(save_dir_feat, 'feat_' + str(batch) +'.csv'), feat_arr, delimiter=",")
        np.savetxt(os.path.join(save_dir_label, 'label_' + str(batch) +'.csv'), label_arr, delimiter=",")

        print( " [Batch %d/%d]  " % ( batch+1, len(loader)),  end = '\r' )


if __name__=='__main__':
    args = parser.arg_parse()
    P3 = P3(args, device)

    
    train_data_dir = '../hw4_data/FullLengthVideos/videos/train'
    train_label_dir = '../hw4_data/FullLengthVideos/labels/train'
    train_folders = os.listdir(train_data_dir)
    train_folders.sort()
    for folder in train_folders:
        print('===> prepare dataloader ...')
        train_loader = torch.utils.data.DataLoader(
            data.DATA(os.path.join(train_data_dir, folder), os.path.join(train_label_dir, folder + '.txt')), 
            batch_size=256, num_workers=args.workers, shuffle=False
        )
        save_dir_feat = os.path.join('train_feature', folder)
        save_dir_label = os.path.join('train_labels', folder)
        feat_extract(P3, train_loader, device, save_dir_feat, save_dir_label)

    valid_data_dir = '../hw4_data/FullLengthVideos/videos/valid'
    valid_label_dir = '../hw4_data/FullLengthVideos/labels/valid'
    valid_folders = os.listdir(valid_data_dir)
    valid_folders.sort()
    for folder in valid_folders:
        print('===> prepare dataloader ...')
        valid_loader = torch.utils.data.DataLoader(
            data.DATA(os.path.join(valid_data_dir, folder), os.path.join(valid_label_dir, folder + '.txt')), 
            batch_size=256, num_workers=args.workers, shuffle=False
        )
        save_dir_feat = os.path.join('valid_feature', folder)
        save_dir_label = os.path.join('valid_labels', folder)
        feat_extract(P3, valid_loader, device, save_dir_feat, save_dir_label)

    
    

    

