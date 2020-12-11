import os
import torch

import numpy as np

import data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision


from models import f_ext, label_cl, domain_cl
from GradReverse import grad_reverse

from PIL import Image

class DANN(nn.Module):
    def __init__(self, args):
        super(DANN, self).__init__()
        ''' load dataset and prepare data loader '''
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.resume_pth = args.resume_pth
        # self.resume_f = args.resume_f
        # self.resume_l = args.resume_l
        # self.resume_d = args.resume_d
        ''' load model '''
        print('===> prepare model ...')
        self.f_ext = f_ext()
        self.label_cl = label_cl()
        self.domain_cl = domain_cl()

        ''' define loss '''
        self.bce_loss = torch.nn.BCELoss()
        self.ce_loss = torch.nn.CrossEntropyLoss()
        ''' setup optimizer '''
        self.opt_f = torch.optim.Adam(self.f_ext.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
        self.opt_l = torch.optim.Adam(self.label_cl.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
        self.opt_d = torch.optim.Adam(self.domain_cl.parameters(), lr=args.lr, weight_decay=args.weight_decay) 

    # def forward(self, x):
    #     feature = self.f_ext(x)
    #     feature = feature.view(feature.shape[0], -1)
    #     label_out = self.label_cl(feature)
    #     domain_out = self.domain_cl(feature)

    #     return label_out, domain_out

    def forward(self, x, alpha):
        feature = self.f_ext(x)
        feature = feature.view(feature.shape[0], -1)
        rev_feature = grad_reverse(feature, alpha)
        label_out = self.label_cl(feature)
        domain_out = self.domain_cl(rev_feature)

        return label_out, domain_out

    def feature(self, x):
        feature = self.f_ext(x)
        feature = feature.view(feature.shape[0], -1)

        return feature

    def save_model(self, epoch):
        if epoch == -1:
            torch.save(self.f_ext.state_dict(), os.path.join(self.save_dir, 'model/', 'Best_F_ext.pth'))  
            torch.save(self.domain_cl.state_dict(), os.path.join(self.save_dir, 'model/', 'Best_D_cl.pth'))  
            torch.save(self.label_cl.state_dict(), os.path.join(self.save_dir, 'model/', 'Best_L_cl.pth'))  
        else: 
            torch.save(self.f_ext.state_dict(), os.path.join(self.save_dir, 'model/', str(epoch) + 'F_ext.pth'))  
            torch.save(self.domain_cl.state_dict(), os.path.join(self.save_dir, 'model/', str(epoch) + 'D_cl.pth'))  
            torch.save(self.label_cl.state_dict(), os.path.join(self.save_dir, 'model/', str(epoch) + 'L_cl.pth'))  

    def load_model(self):

        checkpoint_f = torch.load(os.path.join(self.resume_pth, 'model/', 'Best_F_ext.pth'))
        self.f_ext.load_state_dict(checkpoint_f)
        print("f loaded, " + str(os.path.join(self.resume_pth, 'model/', 'Best_F_ext.pth')))

        checkpoint_l = torch.load(os.path.join(self.resume_pth, 'model/', 'Best_L_cl.pth'))
        self.label_cl.load_state_dict(checkpoint_l)
        print("l loaded, " + str(os.path.join(self.resume_pth, 'model/', 'Best_L_cl.pth')))

        checkpoint_d = torch.load(os.path.join(self.resume_pth, 'model/', 'Best_D_cl.pth'))
        self.domain_cl.load_state_dict(checkpoint_d)
        print("d loaded, " + str(os.path.join(self.resume_pth, 'model/', 'Best_D_cl.pth')))

    def load_test_model(self, f_path, l_path, device):

        checkpoint_f = torch.load(f_path, map_location = device)
        self.f_ext.load_state_dict(checkpoint_f)
        print("f loaded, " + str(f_path))

        checkpoint_l = torch.load(l_path, map_location = device)
        self.label_cl.load_state_dict(checkpoint_l)
        print("l loaded, " + str(l_path))

