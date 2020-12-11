import os
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import cv2
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
 
if __name__ == '__main__':
      

    label_dir = 'hw4_data/FullLengthVideos/labels/valid/OP04-R04-ContinentalBreakfast.txt'
    pred_dir = 'OP04-R04-ContinentalBreakfast.txt'

    label_arr = np.genfromtxt(label_dir)
    pred_arr = np.genfromtxt(pred_dir)



    colors = ['xkcd:blue', 'xkcd:orange', 'xkcd:green', 'xkcd:red', 'xkcd:purple', 'xkcd:brown', 'xkcd:pink', 'xkcd:gray', 'xkcd:olive', 'xkcd:cyan', 'xkcd:yellow']
    y_gt = 0
    y_pred = 15
    step = 10
    height = 3
    x = []
    colors_gt = []
    colors_pred = []
    for i in range(label_arr.shape[0]):
        x.append((i*step, step))
        colors_gt.append(colors[int(label_arr[i])])
        colors_pred.append(colors[int(pred_arr[i])])

    plt.figure(figsize=(18,3))
    plt.broken_barh(x, (y_gt, height), color = colors_gt)
    plt.broken_barh(x, (y_pred, height), color = colors_pred)

    plt.savefig('p3_visual.png')


