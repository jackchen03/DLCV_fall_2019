import os
import torch
import numpy as np

import parser
import data_feat
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from P3 import P3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, train_dir, device):
    epoch_loss = 0
    iteration = 0
    train_folders = os.listdir(train_dir)
    train_folders.sort()
    for folder in train_folders:
        # print('===> prepare dataloader ...')
        feature_dir = os.path.join('train_feature', folder)
        label_dir = os.path.join('train_labels', folder)
        train_loader = torch.utils.data.DataLoader(
            data_feat.DATA(feature_dir, label_dir), 
            batch_size=1, num_workers=args.workers, shuffle=False
        )

        for batch, (x, label) in enumerate(train_loader):
            x = x.to(device)
            label = label.to(device)

            loss = model.update(x.float(), label)
            iteration += 1
            epoch_loss += loss
            print( " [Batch %d/%d] [loss: %f] " % ( batch+1, len(train_loader), loss.item()),  end = '\r' )   

    return epoch_loss / iteration

def eval(model, valid_dir, device, phase):
    total_num = 0
    acc_num = 0

    valid_folders = os.listdir(valid_dir)
    valid_folders.sort()
    for folder in valid_folders:
        # print('===> prepare dataloader ...')
        feature_dir = os.path.join( phase + '_feature', folder)
        label_dir = os.path.join( phase + '_labels', folder)
        valid_loader = torch.utils.data.DataLoader(
            data_feat.DATA(feature_dir, label_dir), 
            batch_size=1, num_workers=args.workers, shuffle=False
        )

        for batch, (x, label) in enumerate(valid_loader):
            x = x.to(device)
            label = label.to(device)

            pred = model.forward_classify(x.float())
            pred = pred.reshape(pred.shape[0]*pred.shape[1], -1)  # (bsize*seq_len, 11)
            label = label.reshape(-1)       # (bsize*seq_len)
            _, pred = torch.max(pred, dim = 1)
            for i in range(pred.shape[0]):
                total_num += 1
                if pred[i].item() == label[i].item():
                    acc_num += 1

    return acc_num / total_num


if __name__=='__main__':
    args = parser.arg_parse()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, 'model/') ):
        os.makedirs(os.path.join(args.save_dir, 'model/') )

    P3 = P3(args, device)
    P3.load_hw_model('../p2_cl.pth', '../p2_rnn.pth')

    lr_sch_c = torch.optim.lr_scheduler.MultiStepLR(P3.opt_c, milestones=[2,6,16,25], gamma=0.3)
    lr_sch_r = torch.optim.lr_scheduler.MultiStepLR(P3.opt_rnn, milestones=[2,6,16,25], gamma=0.3)
    
    best_acc = 0
    loss_list = []

    for epoch in range(1, args.epoch+1):
        print('epoch ' + str(epoch) + ' starts.')

        train_dir = os.path.join(args.data_dir, 'train') 
        epoch_loss = train(P3, train_dir, device)
        print('\n')
        print('epoch_loss = ' + str(epoch_loss.item()))

        if epoch % args.val_epoch == 0:
            valid_dir = os.path.join(args.data_dir, 'valid') 
            acc = eval(P3, valid_dir, device, 'valid')
            print('valid acc = ' + str(acc))
            if acc > best_acc:
                P3.save_model(-1)
                best_acc = acc
            # P3.save_model(epoch)

            acc = eval(P3, train_dir, device, 'train')
            print('train acc = ' + str(acc))

        lr_sch_c.step()
        lr_sch_r.step()

    loss_arr = np.asarray(loss_list)
    np.savetxt(os.path.join(args.save_dir, "loss.csv"), loss_arr, delimiter=",")



