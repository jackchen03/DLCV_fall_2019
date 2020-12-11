import os
import scipy.io
import torch.utils.data as data
import torch

import numpy as np
import pandas as pd
from PIL import Image

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
		self.labels = np.genfromtxt(label_dir)
		self.data_dir = data_dir
		self.data_list = os.listdir(data_dir)
		self.data_list.sort()
		self.transform = transforms.Compose([
			transforms.Resize((224, 224)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406],
	                            std=[0.229, 0.224, 0.225])
		])

	def __len__(self):

		return self.labels.shape[0]

	def __getitem__(self, idx):
		frame = Image.open(os.path.join(self.data_dir, self.data_list[idx]))
		frame = self.transform(frame)
		label = int(self.labels[idx])

		return frame, label