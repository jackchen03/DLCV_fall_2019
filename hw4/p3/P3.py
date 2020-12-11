import os
import torch

import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torchvision


from models import f_ext, label_cl, RNN


class P3():
    def __init__(self, args, device, hidden_dim = 512):
        ''' parameters  '''
        self.device = device
        self.save_dir = args.save_dir
        self.resume_pth = args.resume_pth

        ''' load model '''
        print('===> prepare model ...')
        self.f_ext = f_ext().to(self.device)
        if args.bidirectional :
            num_direction = 2
        else: 
            num_direction = 1
        self.label_cl = label_cl(11, num_direction*hidden_dim, args.drop_cl).to(self.device)
        self.rnn = RNN(2048, hidden_dim, args.num_layer, args.bidirectional, self.device, args.drop_rnn).to(self.device)

        ''' loss definition '''
        self.cl_loss = nn.CrossEntropyLoss()

        ''' optimizer '''
        self.opt_c = torch.optim.Adam(self.label_cl.parameters(), lr=args.lr, betas = (0.5,0.99), weight_decay=args.weight_decay)
        self.opt_rnn = torch.optim.Adam(self.rnn.parameters(), lr=args.lr,  betas = (0.5,0.99), weight_decay=args.weight_decay)
        # self.opt_c = torch.optim.SGD(self.label_cl.parameters(), lr=args.lr, momentum = 0.9, weight_decay=args.weight_decay)
        # self.opt_rnn = torch.optim.SGD(self.rnn.parameters(), lr=args.lr,  momentum = 0.9, weight_decay=args.weight_decay)

    def forward_feat(self, x):
        x = self.f_ext(x)
        x = x.reshape(x.shape[0], -1)

        return x

    def forward_classify(self, x):
        x = self.rnn(x)   # output_shape =  bsize X seq_len X hidden_dim
        x = self.label_cl(x)
        return x

    def forward_rnn_feat(self, x):
        x = self.rnn(x)

        return x

    def save_model(self, epoch):    
        if epoch == -1:
            torch.save(self.label_cl.state_dict(), os.path.join(self.save_dir, 'model/', 'Best_CL.pth'))
            torch.save(self.rnn.state_dict(), os.path.join(self.save_dir, 'model/', 'Best_RNN.pth'))
        else: 
            torch.save(self.label_cl.state_dict(), os.path.join(self.save_dir, 'model/', str(epoch) + '_CL.pth'))
            torch.save(self.rnn.state_dict(), os.path.join(self.save_dir, 'model/', str(epoch) + '_RNN.pth')) 


    def load_model(self, epoch):
        if epoch == -1:
            checkpoint_l = torch.load(os.path.join(self.resume_pth, 'model/', 'Best_CL.pth'))
            self.label_cl.load_state_dict(checkpoint_l)
            print("l loaded, " + str(os.path.join(self.resume_pth, 'model/', 'Best_CL.pth')))

            checkpoint_r = torch.load(os.path.join(self.resume_pth, 'model/', 'Best_RNN.pth'))
            self.rnn.load_state_dict(checkpoint_r)
            print("RNN loaded, " + str(os.path.join(self.resume_pth, 'model/', 'Best_RNN.pth')))

        else:
            checkpoint_l = torch.load(os.path.join(self.resume_pth, 'model/', str(epoch) + '_CL.pth'))
            self.label_cl.load_state_dict(checkpoint_l)
            print("l loaded, " + str(os.path.join(self.resume_pth, 'model/', str(epoch) + '_CL.pth')))

            checkpoint_r = torch.load(os.path.join(self.resume_pth, 'model/', str(epoch) + '_RNN.pth'))
            self.rnn.load_state_dict(checkpoint_r)
            print("RNN loaded, " + str(os.path.join(self.resume_pth, 'model/', str(epoch) + '_RNN.pth')))

    def load_hw_model(self, l_dir, r_dir, device):
        checkpoint_l = torch.load(l_dir, map_location = device)
        self.label_cl.load_state_dict(checkpoint_l)
        print("l loaded, " + str(l_dir))

        checkpoint_r = torch.load(r_dir, map_location = device)
        self.rnn.load_state_dict(checkpoint_r)
        print("RNN loaded, " + str(r_dir))


    def update(self, x, label):
        self.opt_c.zero_grad()
        self.opt_rnn.zero_grad()

        pred = self.forward_classify(x)   # (bsize, seq_len, 11)

        pred = pred.reshape(pred.shape[0]*pred.shape[1], -1)  # (bsize*seq_len, 11)
        label = label.reshape(-1)       # (bsize*seq_len)

        loss = self.cl_loss(pred, label.long())
        loss.backward()

        self.opt_c.step()
        self.opt_rnn.step()

        return loss




        

