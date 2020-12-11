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
		df = pd.read_csv(label_dir)
		self.label_list = df['Action_labels']
		self.video_name = df['Video_name']
		self.video_cate = df['Video_category']
		self.data_dir = data_dir
		# self.video_len = video_len
		self.transform = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize((224, 224)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406],
	                            std=[0.229, 0.224, 0.225])
		])

	def __len__(self):

		return len(self.label_list)

	def __getitem__(self, idx):
		video = readShortVideo(self.data_dir, self.video_cate[idx], self.video_name[idx])
		
		first = self.transform(video[0]).unsqueeze(0)
		last = self.transform(video[-1]).unsqueeze(0)
		images = torch.cat((first, last), 0)
		
		label = self.label_list[idx]

		# video = torch.FloatTensor(video)
		# number = [randint(0, video.shape[0]-1) for p in range(0, 2)]
		# video = torch.cat((video[number[0]].unsqueeze(0), video[number[1]].unsqueeze(0)), 0)
		# video = video / 255
		# ones = 0.5*torch.ones(video.shape)
		# video = (video - ones) / ones
		# norm_term = 64*torch.ones(video.shape)
		# video = (video - norm_term) / norm_term

		# video = zero_padding(video, self.video_len)

		return images, label