import os
import scipy.io
import torch.utils.data as data
import torch
import torchvision.transforms as transforms


import numpy as np
import pandas as pd

from reader import readShortVideo

from random import randint

# def zero_padding(video, padding_size):
# 	if video.shape[0] < padding_size:
# 		zero = torch.zeros(padding_size - video.shape[0], video.shape[1], video.shape[2], video.shape[3])
# 		video = torch.cat((video, zero), 0)
# 	if video.shape[0] > padding_size:
# 		video = video[:padding_size]

# 	return video

def zero_padding(feat, padding_size):
    if feat.shape[0] < padding_size:
        zero = torch.zeros(padding_size - feat.shape[0], feat.shape[1])
        feat = torch.cat((feat.float(), zero), 0)
        
    if feat.shape[0] > padding_size:
        feat = feat[:padding_size].float()
    if feat.shape[0] == padding_size:
    	feat = feat.float()

    return feat

class DATA(data.Dataset):
	def __init__(self, data_dir, label_dir):
		self.data_dir = data_dir
		self.data_list = os.listdir(data_dir)
		self.data_list.sort()

		self.label_dir = label_dir
		self.label_list = os.listdir(label_dir)
		self.label_list.sort()


	def __len__(self):

		return len(self.label_list)

	def __getitem__(self, idx):
		feat_arr = np.genfromtxt(os.path.join(self.data_dir, self.data_list[idx]), delimiter=',')
		feat = torch.tensor(feat_arr) 

		label_arr = np.genfromtxt(os.path.join(self.label_dir, self.label_list[idx]), delimiter=',')
		label = torch.tensor(label_arr)
		
		return feat, label