import os
import torch

import sys
sys.path.insert(1, 'ACGAN/')

import parser_acgan
from ACGAN import ACGAN

import sys
sys.path.insert(1, 'GAN/')

import parser_gan
from GAN import GAN

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import cv2

import parser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
	# ACGAN 
    args = parser_acgan.arg_parse()

    model = ACGAN(args, device).to(device)
    g_path = 'ACGAN_G.pth'
    model.load_hw_model(g_path, device)

    save_dir = os.path.join(args.out_path, 'fig2_2.jpg')
    model.test(save_dir)

    # GAN
    args_2 = parser_gan.arg_parse()

    model_2 = GAN(args_2, device).to(device)
    g_path_2 = 'GAN_G.pth'
    model_2.load_hw_model(g_path_2, device)

    save_dir = os.path.join(args.out_path, 'fig1_2.jpg')
    model_2.test(save_dir)

