import os
import torch

import parser
import data

from baseline_model import Net
from unet import UNet
from deeplab import DeepLab
from deeplabv3 import DeepLabv3

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from sklearn.metrics import accuracy_score
import mean_iou_evaluate

from PIL import Image
import cv2


def evaluate(model, data_loader):

    ''' set model to evaluate mode '''
    model.eval()
    preds = []
    gts = []
    with torch.no_grad(): # do not need to caculate information for gradient during eval
        for idx, (imgs,gt) in enumerate(data_loader):
            imgs = imgs.cuda()
            pred = model(imgs)
            
            _, pred = torch.max(pred, dim = 1)
            # _, gt = torch.max(gt, dim = 3)

            pred = pred.cpu().numpy().squeeze()
            gt = gt.numpy().squeeze()
            
            preds.append(pred)
            gts.append(gt)

    gts = np.concatenate(gts)
    preds = np.concatenate(preds)

    return mean_iou_evaluate.mean_iou_score(preds, gts)

if __name__ == '__main__':
    
    args = parser.arg_parse()

    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)

    ''' prepare data_loader '''
    print('===> prepare data loader ...')
    test_loader = torch.utils.data.DataLoader(data.DATA(args, mode='test'),
                                              batch_size=args.test_batch, 
                                              num_workers=args.workers,
                                              shuffle=False)
    ''' model type'''
    if args.model_type == 'baseline':
        model = Net(args).cuda()
    if args.model_type == 'unet':
        model = UNet(3, 9, args.bilinear).cuda()
    if args.model_type == 'deeplab':
        model = DeepLab(args).cuda()
    if args.model_type == 'deeplabv3':
        model = DeepLabv3(args.output_stride).cuda() 

    ''' resume save model '''
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint)

    acc = evaluate(model, test_loader)
    print('Testing Accuracy: {}'.format(acc))
