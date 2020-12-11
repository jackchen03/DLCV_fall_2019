import os
import torch

import parser
import data
import test

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter
from test import evaluate

from baseline_model import Net
from unet import UNet
from deeplab import DeepLab
from deeplabv3 import DeepLabv3

def save_model(model, save_path):
    torch.save(model.state_dict(),save_path)    



if __name__=='__main__':

    args = parser.arg_parse()
    
    '''create directory to save trained model and other info'''
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)
    
    ''' setup random seed '''
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    ''' load dataset and prepare data loader '''
    print('===> prepare dataloader ...')
    train_loader = torch.utils.data.DataLoader(data.DATA(args, mode='train'),
                                               batch_size=args.train_batch, 
                                               num_workers=args.workers,
                                               shuffle=True)
    val_loader   = torch.utils.data.DataLoader(data.DATA(args, mode='val'),
                                               batch_size=args.train_batch, 
                                               num_workers=args.workers,
                                               shuffle=False)
    ''' load model '''
    print('===> prepare model ...')
    if args.model_type == 'unet':
        model = UNet(3, 9, args.bilinear).cuda()
    if args.model_type == 'baseline':
        model = Net(args).cuda()
    if args.model_type == 'deeplab':
        model = DeepLab(args).cuda()
    if args.model_type == 'deeplabv3':
        model = DeepLabv3(args.output_stride).cuda()

    ''' define loss '''
    criterion = nn.CrossEntropyLoss()
    ''' setup optimizer '''
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.5)
    
    ''' setup tensorboard '''
    writer = SummaryWriter(os.path.join(args.save_dir, 'train_info'))

    ''' train model '''
    print('===> start training ...')
    iters = 0
    best_acc = 0
    loss_list_iter = []
    miou_list_epoch = []
    for epoch in range(1, args.epoch+1):
        
        model.train()
        
        for idx, (imgs, seg) in enumerate(train_loader):
            
            train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx+1, len(train_loader))
            iters += 1

            ''' move data to gpu '''
            imgs, seg = imgs.cuda(), seg.cuda()
            
            ''' forward path '''
            output = model(imgs)
            # output = output.permute(0,2,3,1)

            ''' compute loss, backpropagation, update parameters '''
            loss = criterion(output, seg) # compute loss
            
            optimizer.zero_grad()         # set grad of all parameters to zero
            loss.backward()               # compute gradient for each parameters
            optimizer.step()              # update parameters

            ''' write out information to tensorboard '''
            writer.add_scalar('loss', loss.data.cpu().numpy(), iters)
            train_info += ' loss: {:.4f}'.format(loss.data.cpu().numpy())

            print(train_info)

            loss_list_iter.append([iters, loss.data.cpu().numpy()])
        
        if epoch%args.val_epoch == 0:
            ''' evaluate the model '''
            acc = evaluate(model, val_loader)        
            writer.add_scalar('val_acc', acc, iters)
            print('Epoch: [{}] ACC:{}'.format(epoch, acc))

            miou_list_epoch.append([epoch, acc])

            # ''' save best model '''
            # if acc > best_acc:
            #     save_model(model, os.path.join(args.save_dir, 'model_best.pth.tar'))
            #     best_acc = acc

        # scheduler.step()
        # ''' save model '''
        # save_model(model, os.path.join(args.save_dir, 'model_{}.pth.tar'.format(epoch)))

    np.savetxt("loss_iter.csv", np.asarray(loss_list_iter), delimiter=",")
    np.savetxt("miou_epoch.csv", np.asarray(miou_list_epoch), delimiter=",")
