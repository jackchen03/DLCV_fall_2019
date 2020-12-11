import os
import torch
import numpy as np

import parser
import data_feat
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from P2 import P2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, loader, device):
    epoch_loss = 0

    for batch, (x, label, length) in enumerate(loader):
        x = x.to(device)
        label = label.to(device)

        bsize = x.shape[0]
        # h0 = torch.randn(model.rnn.num_direction*model.rnn.num_layers, bsize, model.rnn.hidden_size).to(device)
        # c0 = torch.randn(model.rnn.num_direction*model.rnn.num_layers, bsize, model.rnn.hidden_size).to(device)

        loss = model.update(x.float(), label, length)
        epoch_loss += loss
        print( " [Batch %d/%d] [loss: %f] " % ( batch+1, len(loader), loss.item()),  end = '\r' )

    return epoch_loss / (batch+1)

def eval(model, loader, device, total_num):
    acc_num = 0
    for batch, (x, label, length) in enumerate(loader):
        x = x.to(device)
        label = label.to(device)

        bsize = x.shape[0]
        # h0 = torch.randn(model.rnn.num_direction*model.rnn.num_layers, bsize, model.rnn.hidden_size).to(device)
        # c0 = torch.randn(model.rnn.num_direction*model.rnn.num_layers, bsize, model.rnn.hidden_size).to(device)

        pred = model.forward_classify(x.float(), length)
        _, pred = torch.max(pred, dim = 1)
        for i in range(pred.shape[0]):
            if pred[i].item() == label[i].item():
                acc_num += 1

    return acc_num / total_num


if __name__=='__main__':
    args = parser.arg_parse()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, 'model/') ):
        os.makedirs(os.path.join(args.save_dir, 'model/') )

    P2 = P2(args, device)

    lr_sch_c = torch.optim.lr_scheduler.MultiStepLR(P2.opt_c, milestones=[2,4,8,16,25], gamma=0.3)
    lr_sch_r = torch.optim.lr_scheduler.MultiStepLR(P2.opt_rnn, milestones=[2,4,8,16,25], gamma=0.3)
    
    total_num = 769  # number of validation video
    best_acc = 0

    loss_list = []
    val_acc = []
    train_acc = []

    print('===> prepare dataloader ...')    
    val_loader = torch.utils.data.DataLoader(
        data_feat.DATA('csv_152' , 'label_valid.csv', args.video_len, 'valid'), 
        batch_size=args.batch_size, num_workers=args.workers, shuffle=False
    )
    for epoch in range(1, args.epoch+1):
        print('epoch ' + str(epoch) + ' starts.')
        number = (epoch - 1) % 12
        feat_dir = 'csv_' + str(number)

        print('===> prepare dataloader ...')
        train_loader = torch.utils.data.DataLoader(
            data_feat.DATA(feat_dir, 'label_train.csv', args.video_len, 'train'), 
            batch_size=args.batch_size, num_workers=args.workers, shuffle=False
        )

        epoch_loss = train(P2, train_loader, device)
        print('epoch loss = ' + str(epoch_loss))
        loss_list.append([epoch, epoch_loss])

        if epoch % args.val_epoch == 0:
            acc = eval(P2, val_loader, device, total_num)
            print('valid acc = ' + str(acc))
            # val_acc.append([epoch, val_acc])
            if acc > best_acc:
                P2.save_model(-1)
                best_acc = acc
            P2.save_model(epoch)

            acc = eval(P2, train_loader, device, 2653)
            print('train acc = ' + str(acc))
            # train_acc.append([epoch, train_acc])

        lr_sch_c.step()
        lr_sch_r.step()

    loss_arr = np.asarray(loss_list)
    np.savetxt(os.path.join(args.save_dir, "loss.csv"), loss_arr, delimiter=",")

    # train_acc = np.asarray(train_acc)
    # np.savetxt(os.path.join(args.save_dir, "train_acc.csv"), train_acc, delimiter=",")
    # val_acc = np.asarray(val_acc)
    # np.savetxt(os.path.join(args.save_dir, "valid_acc.csv"), val_acc, delimiter=",")


