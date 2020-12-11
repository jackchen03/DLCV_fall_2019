import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class f_ext(nn.Module):
	def __init__(self):
		super(f_ext, self).__init__()
		model_ft = models.resnet50(pretrained=True)
		self.resnet_feat = torch.nn.Sequential(*list(model_ft.children())[:-1])

	def forward(self, x):
		with torch.no_grad():
			x = self.resnet_feat(x)

		return x

# class label_cl(nn.Module):
# 	def __init__(self, num_classes):
# 		super(label_cl, self).__init__()
# 		self.hidden1 = nn.Linear(2048*2, num_classes)
# 		# self.hidden2 = nn.Linear(1024, 256)
# 		# self.hidden2 = nn.Linear(512, num_classes)
# 		# self.batch = nn.BatchNorm1d(512)
# 		self.dropout = nn.Dropout(0.4)

# 	def forward(self, x):
# 		x = F.relu(self.dropout(self.hidden1(x)))
# 		# x = F.relu(self.hidden2(x))
# 		# x = F.relu(self.hidden3(x))

# 		return x


class label_cl(nn.Module):
	def __init__(self, num_classes):
		super(label_cl, self).__init__()
		self.hidden1 = nn.Linear(2048*2, 512)
		self.hidden2 = nn.Linear(512, 512)
		self.hidden3 = nn.Linear(512, num_classes)
		# self.batch = nn.BatchNorm1d(512)
		self.dropout1 = nn.Dropout(0.4)
		self.dropout2 = nn.Dropout(0.4)

	def forward(self, x):
		x = F.relu(self.dropout1(self.hidden1(x)))
		x = F.relu(self.dropout2(self.hidden2(x)))
		x = self.hidden3(x)

		return x