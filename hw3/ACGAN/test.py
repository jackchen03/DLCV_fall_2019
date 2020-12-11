import os
import torch

import parser_acgan

from ACGAN import ACGAN

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    
    args = parser_acgan.arg_parse()

    model = ACGAN(args, device).to(device)
    model.load_model(args.load_dir, args.load_epoch)
    model.test('test')

