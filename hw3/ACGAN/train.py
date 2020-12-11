import os
import torch
import numpy as np

import parser_acgan
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ACGAN import ACGAN
from data_acgan import DATA


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__=='__main__':

    args = parser_acgan.arg_parse()
    
    '''create directory to save trained model and other info'''
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, 'model/')):
        os.makedirs(os.path.join(args.save_dir, 'model/'))
    if not os.path.exists(os.path.join(args.save_dir, 'img/')):
        os.makedirs(os.path.join(args.save_dir, 'img/'))

    print('===> prepare dataloader ...')
    loader = torch.utils.data.DataLoader(DATA(args),
                                       batch_size=args.batch_size, 
                                       num_workers=args.workers,
                                       shuffle=True)
    acgan = ACGAN(args, device).to(device)
    for i in range(args.epoch):
        acgan.train(loader)
        if (args.epoch+1) % args.val_epoch == 0:
            acgan.save_imgs(i)
            acgan.save_model(i)
            print("model saved.")

