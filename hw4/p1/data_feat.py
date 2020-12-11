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

class DATA(data.Dataset):
	def __init__(self, data_dir, label_dir):
		self.label_list = np.genfromtxt(label_dir, delimiter=',')
		self.feat_list = np.genfromtxt(data_dir, delimiter=',')

	def __len__(self):

		return len(self.label_list)

	def __getitem__(self, idx):
		feat = torch.tensor(self.feat_list[idx]) 
		label = torch.tensor(self.label_list[idx])

		return feat, label