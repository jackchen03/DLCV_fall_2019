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

from DANN import DANN
import data

class tSNE():
	def __init__(self, args, device):
		self.data_dir = '../hw3_data/digits'
		self.device = device
		self.DANN = DANN(args).to(self.device)
		self.DANN.load_model()
		self.tran_di = args.tran_di

		self.features = []
		self.labels = []

		sv_loader = torch.utils.data.DataLoader(data.DATA(args, 'test', 'svhn'),
										batch_size=args.batch_size, 
										num_workers=args.workers,
										shuffle=False)
		mn_loader = torch.utils.data.DataLoader(data.DATA(args, 'test', 'mnistm'),
										batch_size=args.batch_size, 
										num_workers=args.workers,
										shuffle=False)

		for batch, (sv_imgs, sv_labels, _) in enumerate(sv_loader):
			print(batch, end= '\r')
			sv_imgs = sv_imgs.to(self.device)
			feature = self.DANN.feature(sv_imgs)
			feature = feature.cpu().detach().numpy()
			for i in range(feature.shape[0]):
				self.features.append(feature[i])
				self.labels.append(sv_labels[i].item())

		for batch, (mn_imgs, mn_labels, _) in enumerate(mn_loader):
			print(batch, end= '\r')
			mn_imgs = mn_imgs.to(self.device)
			feature = self.DANN.feature(mn_imgs)
			feature = feature.cpu().detach().numpy()
			for i in range(feature.shape[0]):
				self.features.append(feature[i])
				self.labels.append(mn_labels[i].item())

		sv_domain = np.zeros(26032)
		mn_domain = np.ones(10000)

		self.domain = np.concatenate((sv_domain, mn_domain), 0)
		self.features = np.asarray(self.features).reshape(36032, -1)
		self.labels = np.asarray(self.labels)


	def compute_tsne(self):
		self.tsne_features = sklearn.manifold.TSNE(n_components=2).fit_transform(self.features)
		print("tsne computed.")
		if self.tran_di == 'sv_mn':
			np.savetxt("tsne_feature_svmn.csv", self.tsne_features, delimiter=",")
		if self.tran_di == 'mn_sv':
			np.savetxt("tsne_feature_mnsv.csv", self.tsne_features, delimiter=",")

	def load_tsne(self):
		if self.tran_di == 'sv_mn':
			self.tsne_features = np.genfromtxt("tsne_feature_svmn.csv", delimiter=",")
		if self.tran_di == 'mn_sv':
			self.tsne_features = np.genfromtxt("tsne_feature_mnsv.csv", delimiter=",")

		print("Csv loaded.")

	def draw_domain(self):
		sv_digits = self.tsne_features[:26032]
		mn_digits = self.tsne_features[26032:]

		plt.scatter(sv_digits[:,0], sv_digits[:,1], s=1, marker='x', c='tab:blue')
		plt.scatter(mn_digits[:,0], mn_digits[:,1], s=1, marker='o', c='tab:red')

		if self.tran_di == 'sv_mn':
			plt.savefig("tsne_domain_svmn"+".png")
		if self.tran_di == 'mn_sv':
			plt.savefig("tsne_domain_mnsv"+".png")


	def classify(self):
		self.digit_feature = []
		for i in range(10):
			tmp = []
			self.digit_feature.append(tmp)

		for j in range(self.labels.shape[0]):
			number = self.labels[j]
			self.digit_feature[number].append(self.tsne_features[j])

		for k in range(10):
			self.digit_feature[k] = np.asarray(self.digit_feature[k])

		print(len(self.digit_feature))
		print(self.digit_feature[0].shape)
		print(self.digit_feature[1].shape)
		print("classify completed.")
	def draw_class(self):
		colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

		for i in range(10):
			plt.scatter(self.digit_feature[i][:,0], self.digit_feature[i][:,1], s=1, marker='x', c=colors[i])

		plt.savefig("tsne_class_" + str(self.tran_di) + ".png")
		print("class pic saved.")

	def draw(self):
		# self.compute_tsne()
		self.load_tsne()
		self.draw_domain()
		self.classify()
		self.draw_class()
