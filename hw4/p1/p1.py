import os
import torch

import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torchvision


from models import f_ext, label_cl


class P1():
    def __init__(self, args, device):
        ''' parameters  '''
        self.device = device
        self.save_dir = args.save_dir
        self.resume_pth = args.resume_pth

        ''' load model '''
        print('===> prepare model ...')
        self.f_ext = f_ext().to(self.device)
        self.label_cl = label_cl(11).to(self.device)
        self.label_cl.apply(self.init_weights)

        ''' loss definition '''
        self.cl_loss = nn.CrossEntropyLoss()

        ''' optimizer '''
        self.opt_c = torch.optim.SGD(self.label_cl.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum = 0.9)

    def forward_feat(self, x):
        pseudo_bsize = x.shape[0]
        x = x.reshape(-1, 3, 224, 224)
        x  = self.f_ext(x)
        x = x.reshape(pseudo_bsize, -1)
        return x

    def forward_classify(self, x):
        return self.label_cl(x)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            # m.weight.data.fill_(0.001)
            m.bias.data.fill_(0.01)

    def save_model(self, epoch):    
        if epoch == -1:
            torch.save(self.label_cl.state_dict(), os.path.join(self.save_dir, 'model/', 'Best_CL.pth'))
        else: 
            torch.save(self.label_cl.state_dict(), os.path.join(self.save_dir, 'model/', str(epoch) + '_CL.pth')) 


    def load_model(self, epoch):
        if epoch == -1:
            checkpoint_l = torch.load(os.path.join(self.resume_pth, 'model/', 'Best_CL.pth'))
            self.label_cl.load_state_dict(checkpoint_l)
            print("l loaded, " + str(os.path.join(self.resume_pth, 'model/', 'Best_CL.pth')))

        else:
            checkpoint_l = torch.load(os.path.join(self.resume_pth, 'model/', str(epoch) + '_CL.pth'))
            self.label_cl.load_state_dict(checkpoint_l)
            print("l loaded, " + str(os.path.join(self.resume_pth, 'model/', str(epoch) + '_CL.pth')))

    def load_hw_model(self, l_dir, device):
        checkpoint_l = torch.load(l_dir, map_location = device)
        self.label_cl.load_state_dict(checkpoint_l)
        print("l loaded, " + str(l_dir))

    def update(self, x, label):
        self.opt_c.zero_grad()

        pred = self.forward_classify(x)
        loss = self.cl_loss(pred, label.long())
        loss.backward()

        self.opt_c.step()

        return loss




        

