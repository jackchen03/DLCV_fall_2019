import os
import torch

import numpy as np

import data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision


from models import source_cnn, target_cnn, label_cl, domain_cl

from PIL import Image

class ADDA(nn.Module):
    def __init__(self, args, device):
        super(ADDA, self).__init__()
        ''' load dataset and prepare data loader '''
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.load_dir = args.resume_pth
        self.test_pth = args.test_pth

        self.tran_di = args.tran_di
        self.device = device
        ''' load model '''
        print('===> prepare model ...')
        self.source_cnn = source_cnn()
        self.target_cnn = target_cnn()
        self.label_cl = label_cl()
        self.domain_cl = domain_cl()

        ''' define loss '''
        self.bce_loss = torch.nn.BCELoss()
        self.ce_loss = torch.nn.CrossEntropyLoss()
        
        ''' setup optimizer '''
        self.opt_s = torch.optim.Adam(self.source_cnn.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
        self.opt_t = torch.optim.Adam(self.target_cnn.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
        self.opt_l = torch.optim.Adam(self.label_cl.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
        self.opt_d = torch.optim.Adam(self.domain_cl.parameters(), lr=args.lr, weight_decay=args.weight_decay) 

    # def forward(self, x):
    #     feature = self.f_ext(x)
    #     feature = feature.view(feature.shape[0], -1)
    #     label_out = self.label_cl(feature)
    #     domain_out = self.domain_cl(feature)

    #     return label_out, domain_out

    def forward_source(self, x):
        feature = self.source_cnn(x)
        feature = feature.view(feature.shape[0], -1)
        label_out = self.label_cl(feature)

        return label_out

    def forward_target(self, x):
        feature = self.target_cnn(x)
        feature = feature.view(feature.shape[0], -1)
        label_out = self.label_cl(feature)

        return label_out

    def source_feature(self, x):
        feature = self.source_cnn(x)
        feature = feature.view(feature.shape[0], -1)

        return feature

    def target_feature(self, x):
        feature = self.target_cnn(x)
        feature = feature.view(feature.shape[0], -1)

        return feature

    def update_t_cnn(self, x):
        self.opt_t.zero_grad()
        fake = self.target_feature(x)

        valid = torch.ones(self.batch_size,1).to(self.device)
        loss = self.bce_loss(self.domain_cl(fake),valid)
        loss.backward()
        self.opt_t.step()

        return loss

    def update_d(self, source_x, target_x):
        self.opt_d.zero_grad()
        source_feat = self.source_feature(source_x)
        target_feat = self.target_feature(target_x).detach()

        valid = torch.ones(self.batch_size,1).to(self.device)
        fake = torch.zeros(self.batch_size,1).to(self.device)
        # print("source score" + str(self.domain_cl(source_feat).mean()))
        # print("target score" + str(self.domain_cl(target_feat).mean()))
        real_loss = self.bce_loss(self.domain_cl(source_feat),valid)
        fake_loss = self.bce_loss(self.domain_cl(target_feat),fake)
        loss = real_loss + fake_loss

        loss.backward()
        self.opt_d.step()

        return loss

    def save_model(self, epoch):
        if epoch == -1:
            torch.save(self.source_cnn.state_dict(), os.path.join(self.save_dir, 'model/', 'Best_SCNN.pth'))  
            torch.save(self.target_cnn.state_dict(), os.path.join(self.save_dir, 'model/', 'Best_TCNN.pth')) 
            torch.save(self.domain_cl.state_dict(), os.path.join(self.save_dir, 'model/', 'Best_D_cl.pth'))  
            torch.save(self.label_cl.state_dict(), os.path.join(self.save_dir, 'model/', 'Best_L_cl.pth'))  
        else: 
            torch.save(self.source_cnn.state_dict(), os.path.join(self.save_dir, 'model/', str(epoch) + '_SCNN.pth'))  
            torch.save(self.target_cnn.state_dict(), os.path.join(self.save_dir, 'model/', str(epoch) + '_TCNN.pth')) 
            torch.save(self.domain_cl.state_dict(), os.path.join(self.save_dir, 'model/', str(epoch) + 'D_cl.pth'))  
            torch.save(self.label_cl.state_dict(), os.path.join(self.save_dir, 'model/', str(epoch) + 'L_cl.pth'))  

    def load_source_model(self):
        checkpoint_s = torch.load(os.path.join(self.load_dir, 'model/Best_SCNN.pth'))
        self.source_cnn.load_state_dict(checkpoint_s)
        print("s loaded, " + str(os.path.join(self.load_dir, 'model/Best_SCNN.pth')))

        self.target_cnn.load_state_dict(checkpoint_s)
        print("t loaded, " + str(os.path.join(self.load_dir, 'model/Best_SCNN.pth')))

        checkpoint_l = torch.load(os.path.join(self.load_dir, 'model/Best_L_cl.pth'))
        self.label_cl.load_state_dict(checkpoint_l)
        print("l loaded, " + str(os.path.join(self.load_dir, 'model/Best_L_cl.pth')))

        # checkpoint_d = torch.load(self.resume_d)
        # self.domain_cl.load_state_dict(checkpoint_d)
        # print("d loaded, " + str(self.resume_d))

    def load_model(self):
        checkpoint_t = torch.load(os.path.join(self.test_pth, 'model/Best_TCNN.pth'))
        self.target_cnn.load_state_dict(checkpoint_t)
        print("t loaded, " + str(os.path.join(self.test_pth, 'model/Best_STNN.pth')))

        checkpoint_l = torch.load(os.path.join(self.test_pth, 'model/Best_L_cl.pth'))
        self.label_cl.load_state_dict(checkpoint_l)
        print("l loaded, " + str(os.path.join(self.test_pth, 'model/Best_L_cl.pth')))

    def load_test_model(self, t_path, l_path, device):
        
        checkpoint_t = torch.load(t_path, map_location = device)
        self.target_cnn.load_state_dict(checkpoint_t)
        print("t loaded, " + str(t_path))

        checkpoint_l = torch.load(l_path, map_location = device)
        self.label_cl.load_state_dict(checkpoint_l)
        print("l loaded, " + str(l_path))

    def evaluate(self, loader):
        correct_num = 0
        if self.tran_di == 'mn_sv':
            total_num = 6000
        if self.tran_di == 'sv_mn':
            total_num = 5000
        
        for batch, (imgs, labels, domains) in enumerate(loader):
            print( " [Batch %d/%d]  " % ( batch+1, len(loader)), end = '\r' )
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            domains = domains.to(self.device)

            labels_pred = self.forward_target(imgs)
            _, labels_pred = torch.max(labels_pred, dim = 1)

            # labels = labels.float()
            for i in range(labels.shape[0]):
                if labels_pred[i] == labels[i]:
                    correct_num += 1

        return correct_num / total_num

