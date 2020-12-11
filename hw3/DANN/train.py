import os
import torch
import numpy as np

import parser
import data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from DANN import DANN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(args, loader, DANN, domain = 'svhn'):
    correct_num = 0
    if domain == 'svhn':
        total_num = 6000
    if domain == 'mnistm': 
        total_num = 5000
    for batch, (imgs, labels, domains) in enumerate(loader):
        print( " [Batch %d/%d]  " % ( batch+1, len(loader)), end = '\r' )
        imgs = imgs.to(device)
        labels = labels.to(device)
        domains = domains.to(device)

        labels_pred, _  = DANN.forward(imgs, 0)
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
        target_loader = torch.utils.data.DataLoader(data.DATA(args, 'train', 'mnistm'),
                                               batch_size=args.batch_size, 
                                               num_workers=args.workers,
                                               shuffle=True)
        val_loader = torch.utils.data.DataLoader(data.DATA(args, 'val', 'mnistm'),
                                               batch_size=args.batch_size, 
                                               num_workers=args.workers,
                                               shuffle=True)

    if args.tran_di == 'mn_sv':
        source_loader = torch.utils.data.DataLoader(data.DATA(args, 'train', 'mnistm'),
                                               batch_size=args.batch_size, 
                                               num_workers=args.workers,
                                               shuffle=True)
        target_loader = torch.utils.data.DataLoader(data.DATA(args, 'train', 'svhn'),
                                               batch_size=args.batch_size, 
                                               num_workers=args.workers,
                                               shuffle=True)
        val_loader = torch.utils.data.DataLoader(data.DATA(args, 'val', 'svhn'),
                                               batch_size=args.batch_size, 
                                               num_workers=args.workers,
                                               shuffle=True)


    DANN = DANN(args).to(device)
    if args.resume > 0:
        DANN.load_model()

    best_acc = 0
    acc_list = []

    start_epoch = max(args.resume, 0)
    print(start_epoch)
    for i in range(start_epoch, args.epoch):
        len_dataloader  = min (len(source_loader), len(target_loader))

        for j in range(len_dataloader):
            p = float(j + i * len_dataloader ) / args.epoch / len_dataloader 
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            source_iter = iter(source_loader)
            target_iter = iter(target_loader)

            data_source = source_iter.next()
            source_imgs, source_labels, source_domains = data_source

            data_target = target_iter.next()
            target_imgs, _, target_domains = data_target

            DANN.opt_d.zero_grad()
            DANN.opt_l.zero_grad()
            DANN.opt_f.zero_grad()


            source_imgs = source_imgs.to(device)
            source_labels = source_labels.to(device)
            source_domains = source_domains.to(device)

            s_label_pre, s_domain_pre = DANN(source_imgs, alpha)
            # print(s_domain_pre.mean())

            _, labels_pred = torch.max(s_label_pre, dim = 1)
            
            loss_l = DANN.ce_loss(s_label_pre, source_labels)
            loss_d_s = DANN.bce_loss(s_domain_pre, source_domains)

            target_imgs = target_imgs.to(device)
            target_domains = target_domains.to(device)

            _, t_domain_pre = DANN(target_imgs, alpha)

            loss_d_t = DANN.bce_loss(t_domain_pre, target_domains)

            loss = loss_l + loss_d_s + loss_d_t
            loss.backward()

            DANN.opt_d.step()
            DANN.opt_f.step()
            DANN.opt_l.step()            
            print( " [Batch %d/%d] [loss_l: %f] [loss_d_s: %f] [loss_d_t: %f]" % ( j+1, len(target_loader), loss_l.item(), loss_d_s.item(), loss_d_t.item()), end = '\r' )

        if (args.epoch+1) % args.val_epoch == 0:
            # gan.save_imgs(args, i)

            acc = evaluate(args, val_loader, DANN)
            acc_list.append([i, acc])

            ''' save best model '''
            if acc > best_acc:
                DANN.save_model(-1)
                best_acc = acc
                print("Outperform best model. New best model saved.")

            print("epoch " + str(i+1) + " valid acc = " + str(acc))
            DANN.save_model(i+1)
            print("model saved.")

    np.savetxt("log/acc.csv", np.asarray(acc_list), delimiter=",")
