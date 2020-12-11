import os
import torch

import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision


from gan_model import Generator, Discriminator
from PIL import Image

class GAN(nn.Module):
    def __init__(self, args, device):
        super(GAN, self).__init__()
        ''' load dataset and prepare data loader '''
        self.batch_size = args.batch_size
        self.lamda_gp = args.lamda_gp
        self.g_iter = args.g_iter
        self.save_dir = args.save_dir
        self.device = device

        self.random_seed = args.random_seed

        ''' load model '''
        print('===> prepare GAN model ...')
        self.G = Generator()
        self.D = Discriminator()

        ''' define loss '''
        self.adversarial_loss = torch.nn.BCELoss()
        ''' setup optimizer '''
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=args.lr, weight_decay=args.weight_decay) 

    def update_g(self):
        self.G.zero_grad()
        noise = torch.randn(self.batch_size, 100).to(self.device)
        fake = self.G(noise)
        loss = self.D(fake)
        loss = (-1)*loss.mean()

        valid = torch.ones(self.batch_size,1).to(self.device)
        loss = self.adversarial_loss(self.D(fake),valid)
        loss.backward()
        self.opt_G.step()

        return loss

    def update_d(self, real_img):
        self.D.zero_grad()
        noise = torch.randn(self.batch_size, 100).to(self.device)
        fake_img = self.G(noise).detach()

        valid = torch.ones(self.batch_size,1).to(self.device)
        fake = torch.zeros(self.batch_size,1).to(self.device)
        real_loss = self.adversarial_loss(self.D(real_img),valid)
        fake_loss = self.adversarial_loss(self.D(fake_img),fake)
        loss = real_loss + fake_loss

        loss.backward()
        self.opt_D.step()

        return loss

    def train(self, loader):
        for i, imgs in enumerate(loader):
            imgs = imgs.to(self.device)
            loss_d = self.update_d(imgs)
            for j in range(self.g_iter):
                loss_g = self.update_g()

            print(
                " [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % ( i, len(loader), loss_d.item(),  loss_g.item()),
                end = '\r'
            )

    def save_imgs(self, epoch):

        torch.cuda.manual_seed(self.random_seed)
        noise = torch.randn(32, 100).to(self.device)
        # gen_imgs should be shape (25, 64, 64, 3)
        gen_imgs = self.G(noise).permute(0,2,3,1)
        row1 = gen_imgs[0:8].reshape(-1,64,3)
        row2 = gen_imgs[8:16].reshape(-1,64,3)
        row3 = gen_imgs[16:24].reshape(-1,64,3)
        row4 = gen_imgs[24:32].reshape(-1,64,3)
        print(row1.shape)
        cat = torch.cat((row1,row2,row3,row4),1)
        print(cat.shape)
        cat = 256*(cat*0.5+0.5).cpu().detach().numpy()
        cat = cat.astype(np.uint8)
        print(cat.dtype)
        img = Image.fromarray(cat)
        img.save(os.path.join(self.save_dir, 'img/', str(epoch) + '.png'))

    def save_model(self, epoch):
        torch.save(self.G.state_dict(), os.path.join(self.save_dir, 'model/', str(epoch) + 'G.pth'))  
        torch.save(self.D.state_dict(), os.path.join(self.save_dir, 'model/', str(epoch) + 'D.pth'))  

    def load_model(self, load_dir, epoch):
        d_path = os.path.join(load_dir,  str(epoch) + 'D.pth')
        g_path = os.path.join(load_dir,  str(epoch) + 'G.pth')
        checkpoint_d = torch.load(d_path)
        self.D.load_state_dict(checkpoint_d)
        checkpoint_g = torch.load(g_path)
        self.G.load_state_dict(checkpoint_g)

    def load_hw_model(self, g_path, device):

        checkpoint_g = torch.load(g_path, map_location = device)
        self.G.load_state_dict(checkpoint_g)

    def test(self, out_path):

        torch.cuda.manual_seed(995)
        torch.manual_seed(995)
        noise = torch.randn(32, 100).to(self.device)
        # gen_imgs should be shape (25, 64, 64, 3)
        gen_imgs = self.G(noise).permute(0,2,3,1)
        row1 = gen_imgs[0:8].reshape(-1,64,3)
        row2 = gen_imgs[8:16].reshape(-1,64,3)
        row3 = gen_imgs[16:24].reshape(-1,64,3)
        row4 = gen_imgs[24:32].reshape(-1,64,3)
        cat = torch.cat((row1,row2,row3,row4),1)
        cat = 256*(cat*0.5+0.5).cpu().detach().numpy()
        cat = cat.astype(np.uint8)
        img = Image.fromarray(cat)
        img.save(out_path)
