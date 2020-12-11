import os
import torch
import numpy as np

import parser_gan

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from GAN import GAN
from data_gan import DATA


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__=='__main__':

    args = parser_gan.arg_parse()
    
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

    gan = GAN(args, device).to(device)
    for i in range(args.epoch):
        gan.train(loader)
        if (args.epoch+1) % args.val_epoch == 0:
            # gan.save_imgs(args, i)
            gan.save_model(i)
            print("model saved.")

