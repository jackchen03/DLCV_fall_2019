import os
import torch

import numpy as np

from PIL import Image

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision


from model_acgan import Generator, Discriminator


class ACGAN(nn.Module):
    def __init__(self, args, device):
        super(ACGAN, self).__init__()
        ''' load dataset and prepare data loader '''
        self.batch_size = args.batch_size
        self.lamda_gc = args.lamda_gc
        self.lamda_dc = args.lamda_dc
        self.g_iter = args.g_iter
        self.save_dir = args.save_dir
        self.random_seed = args.random_seed
        self.device = device
        ''' load model '''
        print('===> prepare ACGAN model ...')
        self.G = Generator(1)
        self.D = Discriminator(1)

        ''' define loss '''
        self.adversarial_loss = torch.nn.BCELoss()
        self.class_loss = torch.nn.BCELoss()
        ''' setup optimizer '''
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=args.lr, weight_decay=args.weight_decay) 

    def update_g(self):
        self.G.zero_grad()
        noise = torch.randn(self.batch_size, 100).to(self.device)
        class_labels = torch.FloatTensor(np.random.randint(0, 2, self.batch_size)).reshape(-1,1).to(self.device)
        cat = torch.cat((noise, class_labels), 1)
        fake = self.G(cat)
        # if self.gan_type == 'wgan-gp':
        #     loss = self.D(fake)
        #     loss = (-1)*loss.mean()

        valid = torch.ones(self.batch_size,1).to(self.device)
        tf, cl = self.D(fake)
        loss = self.adversarial_loss(tf,valid) + self.class_loss(cl, class_labels) * self.lamda_gc

        loss.backward()
        self.opt_G.step()

        return self.adversarial_loss(tf,valid), self.class_loss(cl, class_labels) * self.lamda_gc

    def update_d(self, real_img, class_labels):
        self.D.zero_grad()

        noise = torch.randn(self.batch_size, 100).to(self.device)
        cat = torch.cat((noise, class_labels), 1)
        fake_img = self.G(cat).detach()
        # if self.gan_type == 'wgan-gp':
        #     real_loss = (-1)*self.D(real_img).mean()
        #     fake_loss = self.D(fake_img).mean()
        #     gp = self.gradient_penalty(self.D, real_img, fake_img)
        #     loss = real_loss + fake_loss + self.lamda_gp*gp

        valid = torch.ones(self.batch_size,1).to(self.device)
        fake = torch.zeros(self.batch_size,1).to(self.device)
        real_dis, real_cl = self.D(real_img)
        fake_dis, fake_cl = self.D(fake_img)
        real_adv_loss = self.adversarial_loss(real_dis, valid)
        fake_adv_loss = self.adversarial_loss(fake_dis, fake)

        real_cl_loss = self.class_loss(real_cl, class_labels)
        fake_cl_loss = self.class_loss(fake_cl, class_labels)
        loss = real_adv_loss + fake_adv_loss + self.lamda_dc * (real_cl_loss + fake_cl_loss) 

        loss.backward()
        self.opt_D.step()

        return real_adv_loss + fake_adv_loss, self.lamda_dc * (real_cl_loss + fake_cl_loss) 

    def train(self, loader):
        for i, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            loss_d_adv, loss_d_cl = self.update_d(imgs, labels)
            for j in range(self.g_iter):
                loss_g_adv, loss_g_cl = self.update_g()

            print(
                " [Batch %d/%d] [D loss: %f] [D_cl loss: %f] [G loss: %f] [G_cl loss: %f]"
                % ( i, len(loader), loss_d_adv.item(), loss_d_cl.item(),  loss_g_adv.item(), loss_g_cl.item()),
                end = '\r'
            )


    def save_imgs(self, epoch):

        torch.cuda.manual_seed(self.random_seed)      
        noise = torch.randn(32, 100)

        labels_1 = torch.FloatTensor([0])
        labels_2 = torch.FloatTensor([1])
        labels_1 = labels_1.repeat(32, 1)
        labels_2 = labels_2.repeat(32, 1)

        cat_1 = torch.cat((noise, labels_1), 1).to(self.device)
        cat_2 = torch.cat((noise, labels_2), 1).to(self.device)

        # gen_imgs should be shape (25, 64, 64, 3)
        gen_imgs = self.G(cat_1).permute(0,2,3,1)
        row1 = gen_imgs[0:8].reshape(-1,64,3)
        row2 = gen_imgs[8:16].reshape(-1,64,3)
        row3 = gen_imgs[16:24].reshape(-1,64,3)
        row4 = gen_imgs[24:32].reshape(-1,64,3)
        print(row1.shape)
        cat = torch.cat((row1,row2,row3,row4),1)
        print(cat.shape)
        cat = 256*(cat*0.5+0.5).cpu().detach().numpy()
        cat = cat.astype(np.uint8)
        print(str(os.path.join(self.save_dir, 'img/', str(epoch) + '_no_smile.png')) + " saved.")
        img = Image.fromarray(cat)
        img.save(os.path.join(self.save_dir, 'img/', str(epoch) + '_no_smile.png'))

        # gen_imgs should be shape (25, 64, 64, 3)
        gen_imgs = self.G(cat_2).permute(0,2,3,1)
        row1 = gen_imgs[0:8].reshape(-1,64,3)
        row2 = gen_imgs[8:16].reshape(-1,64,3)
        row3 = gen_imgs[16:24].reshape(-1,64,3)
        row4 = gen_imgs[24:32].reshape(-1,64,3)
        print(row1.shape)
        cat = torch.cat((row1,row2,row3,row4),1)
        print(cat.shape)
        cat = 256*(cat*0.5+0.5).cpu().detach().numpy()
        cat = cat.astype(np.uint8)
        print(str(os.path.join(self.save_dir, 'img/', str(epoch) + '_smile.png')) + " saved.")
        img = Image.fromarray(cat)
        img.save(os.path.join(self.save_dir, 'img/', str(epoch) + '_smile.png'))


    def save_model(self, epoch):
        torch.save(self.G.state_dict(), os.path.join(self.save_dir, 'model/', str(epoch) + 'G.pth'))  
        torch.save(self.D.state_dict(), os.path.join(self.save_dir, 'model/', str(epoch) + 'D.pth'))  

    def load_model(self, load_dir, epoch):
        d_path = os.path.join(load_dir, 'model/', str(epoch) + 'D.pth')
        g_path = os.path.join(load_dir, 'model/', str(epoch) + 'G.pth')

        checkpoint_d = torch.load(d_path)
        self.D.load_state_dict(checkpoint_d)
        checkpoint_g = torch.load(g_path)
        self.G.load_state_dict(checkpoint_g)

    def load_hw_model(self, g_path, device):

        checkpoint_g = torch.load(g_path, map_location = device)
        self.G.load_state_dict(checkpoint_g)


    def test(self, out_path):

        torch.cuda.manual_seed(999)
        torch.manual_seed(999)

        noise = torch.randn(10, 100)

        labels_1 = torch.FloatTensor([0])
        labels_2 = torch.FloatTensor([1])
        labels_1 = labels_1.repeat(10, 1)
        labels_2 = labels_2.repeat(10, 1)

        cat_1 = torch.cat((noise, labels_1), 1).to(self.device)
        cat_2 = torch.cat((noise, labels_2), 1).to(self.device)

        # gen_imgs should be shape (25, 64, 64, 3)
        gen_imgs = self.G(cat_1).permute(0,2,3,1)
        row1 = gen_imgs[0:5].reshape(-1,64,3)
        row3 = gen_imgs[5:10].reshape(-1,64,3)

        # gen_imgs should be shape (25, 64, 64, 3)
        gen_imgs = self.G(cat_2).permute(0,2,3,1)
        row2 = gen_imgs[0:5].reshape(-1,64,3)
        row4 = gen_imgs[5:10].reshape(-1,64,3)
        
        cat = torch.cat((row1,row2,row3,row4),1)
        cat = 256*(cat*0.5+0.5).cpu().detach().numpy()
        cat = cat.astype(np.uint8)
        img = Image.fromarray(cat)
        img.save(out_path)
