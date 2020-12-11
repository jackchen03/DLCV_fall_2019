import os
import scipy.io
import torch.utils.data as data
import torch

import numpy as np
import pandas as pd

from reader import readShortVideo

import torchvision.transforms as transforms


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
		self.video_name = df['Video_name']
		self.video_cate = df['Video_category']
		self.data_dir = data_dir
		self.transform = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize((224, 224)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406],
	                            std=[0.229, 0.224, 0.225])
		])

	def __len__(self):

		return len(self.video_name)

	def __getitem__(self, idx):
		video = readShortVideo(self.data_dir, self.video_cate[idx], self.video_name[idx])
		video = torch.FloatTensor(video)

		images = torch.zeros((video.shape[0], video.shape[3], 224, 224))
		for i, frame in enumerate(video):	
			frame = frame.permute(2,0,1)
			image = self.transform(frame)
			images[i] = image
		
		return images