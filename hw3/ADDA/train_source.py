import os
import torch
import numpy as np

import parser
import data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ADDA import ADDA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(args, loader, ADDA, domain = 'svhn'):
    correct_num = 0
    if domain == 'svhn':
        total_num = 6000
    if domain == 'mnistm': 
        total_num = 5000
    for batch, (imgs, labels, domains) in enumerate(loader):
        print( " [Batch %d/%d]  " % ( batch+1, len(loader)), end = '\r' )
        imgs = imgs.to(device)
        labels = labels.to(device)

        labels_pred  = ADDA.forward_source(imgs)
        _, labels_pred = torch.max(labels_pred, dim = 1)

        # labels = labels.float()
        for i in range(labels.shape[0]):
            if labels_pred[i] == labels[i]:
                correct_num += 1

    return correct_num / total_num

if __name__=='__main__':

    args = parser.arg_parse()
    
    '''create directory to save trained model and other info'''
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, 'model/')):
        os.makedirs(os.path.join(args.save_dir, 'model/'))
    if not os.path.exists(os.path.join(args.save_dir, 'img/')):
        os.makedirs(os.path.join(args.save_dir, 'img/'))

    print('===> prepare dataloader ...')
    if args.tran_di == 'sv_mn':
        source_loader = torch.utils.data.DataLoader(data.DATA(args, 'train', 'svhn'),
                                               batch_size=args.batch_size, 
                                               num_workers=args.workers,
                                               shuffle=True)

        val_loader = torch.utils.data.DataLoader(data.DATA(args, 'val', 'svhn'),
                                               batch_size=args.batch_size, 
                                               num_workers=args.workers,
                                               shuffle=True)

    if args.tran_di == 'mn_sv':
        source_loader = torch.utils.data.DataLoader(data.DATA(args, 'train', 'mnistm'),
                                               batch_size=args.batch_size, 
                                               num_workers=args.workers,
                                               shuffle=True)

        val_loader = torch.utils.data.DataLoader(data.DATA(args, 'val', 'mnistm'),
                                               batch_size=args.batch_size, 
                                               num_workers=args.workers,
                                               shuffle=True)


    ADDA = ADDA(args, device).to(device)
    if args.resume > 0:
        ADDA.load_model()

    best_acc = 0
    acc_list = []

    start_epoch = max(args.resume, 0)
    print(start_epoch)

    scheduler_l = torch.optim.lr_scheduler.StepLR(ADDA.opt_l, 15, gamma=0.7, last_epoch=-1)
    scheduler_s = torch.optim.lr_scheduler.StepLR(ADDA.opt_s, 15, gamma=0.7, last_epoch=-1)
    for i in range(start_epoch, args.epoch):
        for batch, (source_imgs, source_labels, source_domains) in enumerate(source_loader):

            ADDA.opt_l.zero_grad()
            ADDA.opt_s.zero_grad()


            source_imgs = source_imgs.to(device)
            source_labels = source_labels.to(device)

            s_label_pre = ADDA.forward_source(source_imgs)
            
            loss = ADDA.ce_loss(s_label_pre, source_labels)
            loss.backward()

            ADDA.opt_s.step()
            ADDA.opt_l.step()            
            print( " [Batch %d/%d] [loss: %f] " % ( batch+1, len(source_loader), loss.item()), end = '\r' )

        if (args.epoch+1) % args.val_epoch == 0:
            # gan.save_imgs(args, i)

            acc = evaluate(args, val_loader, ADDA)
            acc_list.append([i, acc])

            ''' save best model '''
            if acc > best_acc:
                ADDA.save_model(-1)
                best_acc = acc
                print("Outperform best model. New best model saved.")

            print("epoch " + str(i+1) + " valid acc = " + str(acc))
            ADDA.save_model(i+1)
            print("model saved.")

        scheduler_l.step()
        scheduler_s.step()

    np.savetxt( os.path.join(args.save_dir, "acc.csv") , np.asarray(acc_list), delimiter=",")
