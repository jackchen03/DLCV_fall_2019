import os

import numpy as np 
import pandas as pd
from PIL import Image
import sklearn.manifold
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

import data

class tSNE():
	def __init__(self):
		# label_train = np.genfromtxt('label_train.csv', delimiter=',')
		# feat_train = np.genfromtxt('feat_train.csv', delimiter=',')
		self.labels = np.genfromtxt('label_valid.csv', delimiter=',')
		self.features = np.genfromtxt('feat_valid.csv', delimiter=',')

		self.class_num = 11

	def compute_tsne(self):
		self.tsne_features = sklearn.manifold.TSNE(n_components=2).fit_transform(self.features)
		print("tsne computed.")
		np.savetxt("tsne_feature_p1.csv", self.tsne_features, delimiter=",")

	def load_tsne(self):
		self.tsne_features = np.genfromtxt("tsne_feature_p1.csv", delimiter=",")

		print("Csv loaded.")


	def classify(self):
		self.class_feature = []
		for i in range(self.class_num):
			tmp = []
			self.class_feature.append(tmp)

		for j in range(self.labels.shape[0]):
			number = int(self.labels[j])
			self.class_feature[number].append(self.tsne_features[j])

		for k in range(self.class_num):
			self.class_feature[k] = np.asarray(self.class_feature[k])

		print(len(self.class_feature))
		print(self.class_feature[0].shape)
		print(self.class_feature[1].shape)
		print("classify completed.")
	def draw_class(self):
		colors = ['xkcd:blue', 'xkcd:orange', 'xkcd:green', 'xkcd:red', 'xkcd:purple', 'xkcd:brown', 'xkcd:pink', 'xkcd:gray', 'xkcd:olive', 'xkcd:cyan', 'xkcd:yellow']

		for i in range(self.class_num):
			plt.scatter(self.class_feature[i][:,0], self.class_feature[i][:,1], s=1, marker='x', c=colors[i])

		plt.savefig("tsne_p1.png")
		print("class pic saved.")

	def draw(self):
		# self.compute_tsne()
		self.load_tsne()
		self.classify()
		self.draw_class()
