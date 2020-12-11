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


    ADDA = ADDA(args, device).to(device)
        
    ADDA.load_source_model()
    ADDA.source_cnn.eval()
    ADDA.label_cl.eval()

    best_acc = 0
    acc_list = []

    start_epoch = max(args.resume, 0)
    print(start_epoch)
    for i in range(start_epoch, args.epoch):
        len_dataloader  = min (len(source_loader), len(target_loader))

        if i == 0:
            number = args.dis_step
        elif i == 1:
            number = args.dis_step / 5
        elif i < 4:
            number = args.dis_step / 10
        elif i < 6:
            number = args.dis_step / 20
        else:
            number = 0

        for k in range(int(number)):
            print(k, end = '\r')
            source_iter = iter(source_loader)
            target_iter = iter(target_loader)

            data_source = source_iter.next()
            source_imgs, _, _ = data_source

            data_target = target_iter.next()
            target_imgs, _, _ = data_target

            source_imgs = source_imgs.to(device)
            target_imgs = target_imgs.to(device)

            loss_d = ADDA.update_d(source_imgs, target_imgs)

        acc = ADDA.evaluate(val_loader)
        print("After training dis, acc = " + str(acc))
        ADDA.save_model(-1)


        for j in range(len_dataloader):
            source_iter = iter(source_loader)
            target_iter = iter(target_loader)

            data_source = source_iter.next()
            source_imgs, _, _ = data_source

            data_target = target_iter.next()
            target_imgs, _, _ = data_target

            source_imgs = source_imgs.to(device)
            target_imgs = target_imgs.to(device)

            for k in range(args.d_iter):
                loss_d = ADDA.update_d(source_imgs, target_imgs)
            for k in range(args.g_iter):
                loss_g = ADDA.update_t_cnn(target_imgs)

            print(
                " [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % ( j, len_dataloader, loss_d.item(),  loss_g.item()),
                end = '\r'
            )

        if (args.epoch+1) % args.val_epoch == 0:
            # gan.save_imgs(args, i)

            acc = ADDA.evaluate(val_loader)
            acc_list.append([i, acc])

            ''' save best model '''
            if acc > best_acc:
                ADDA.save_model(-1)
                best_acc = acc
                print("Outperform best model. New best model saved.")

            print("epoch " + str(i+1) + " valid acc = " + str(acc))
            ADDA.save_model(i+1)
            print("model saved.")

    np.savetxt("log/acc.csv", np.asarray(acc_list), delimiter=",")
