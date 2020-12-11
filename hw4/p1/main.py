import os
import torch
import numpy as np

import parser
import data_feat
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from p1 import P1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, loader, device):
    epoch_loss = 0
    for batch, (x, label) in enumerate(loader):
        x = x.to(device)
        label = torch.tensor(label)
        label = label.to(device)

        loss = model.update(x.float(), label)
        epoch_loss += loss
        print( " [Batch %d/%d] [loss: %f] " % ( batch+1, len(loader), loss.item()),  end = '\r' )

    return epoch_loss / (batch + 1)

def eval(model, loader, device, total_num):
    acc_num = 0
    for batch, (x, label) in enumerate(loader):
        x = x.to(device)
        label = label.to(device)

        pred = model.forward_classify(x.float())
        _, pred = torch.max(pred, dim = 1)
        for i in range(pred.shape[0]):
            if pred[i].item() == label[i].item():
                acc_num += 1

    return acc_num / total_num



if __name__=='__main__':
    args = parser.arg_parse()
    print('===> prepare dataloader ...')
    train_loader = torch.utils.data.DataLoader(
        data_feat.DATA('feat_train.csv', 'label_train.csv' ), 
        batch_size=args.batch_size, num_workers=args.workers, shuffle=False
    )
    val_loader = torch.utils.data.DataLoader(
        data_feat.DATA('feat_valid.csv', 'label_valid.csv'), 
        batch_size=args.batch_size, num_workers=args.workers, shuffle=False
    )

    P1 = P1(args, device)

    best_acc = 0
    total_num = 769  # number of validation video
    train_num = 2653

    # acc = eval(P1, val_loader, device, total_num)
    # print('acc = ' + str(acc))
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(P1.opt_c, milestones=[2,6,10], gamma=0.32)
    loss_list = []
    for epoch in range(1, args.epoch+1):
        print('epoch ' + str(epoch) + ' starts.')
        epoch_loss = train(P1, train_loader, device)
        loss_list.append([epoch, epoch_loss])

        if epoch % args.val_epoch == 0:
            acc = eval(P1, val_loader, device, total_num)
            print('valid acc = ' + str(acc))
            if acc > best_acc:
                P1.save_model(-1)
                best_acc = acc
            P1.save_model(epoch)
            train_acc = eval(P1, train_loader, device, train_num)
            print('train acc = ' + str(train_acc))

        lr_scheduler.step()

    loss_arr = np.asarray(loss_list)
    np.savetxt(os.path.join(args.save_dir, 'loss.csv'), loss_arr, delimiter=",")

