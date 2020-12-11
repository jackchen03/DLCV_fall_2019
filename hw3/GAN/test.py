import os
import torch

import parser_gan

from GAN import GAN

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# def evaluate(model, data_loader):

#     ''' set model to evaluate mode '''
#     model.eval()
#     preds = []
#     gts = []
#     with torch.no_grad(): # do not need to caculate information for gradient during eval
#         for idx, (imgs,gt) in enumerate(data_loader):
#             imgs = imgs.cuda()
#             pred = model(imgs)
            
#             _, pred = torch.max(pred, dim = 1)
#             # _, gt = torch.max(gt, dim = 3)

#             pred = pred.cpu().numpy().squeeze()
#             gt = gt.numpy().squeeze()
            
#             preds.append(pred)
#             gts.append(gt)

#     gts = np.concatenate(gts)
#     preds = np.concatenate(preds)

#     return mean_iou_evaluate.mean_iou_score(preds, gts)

if __name__ == '__main__':
    
    args = parser_gan.arg_parse()


    ''' model type'''
    if args.model_type == 'baseline':
        model = GAN(args, device).to(device)
    model.load_model(args.load_dir, args.load_epoch)
    model.test('test')

