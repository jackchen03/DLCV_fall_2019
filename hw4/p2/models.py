import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class f_ext(nn.Module):
	def __init__(self):
		super(f_ext, self).__init__()
		model_ft = models.resnet50(pretrained=True)
		self.resnet_feat = torch.nn.Sequential(*list(model_ft.children())[:-1])
		self.resnet_feat.eval()

	def forward(self, x):
		with torch.no_grad():
			x = self.resnet_feat(x)

		return x

class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, bidirectional, device, dropout):
		super(RNN, self).__init__()
		# self.rnn = nn.GRU(input_size, hidden_size, batch_first = False)
		self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, dropout = (dropout if num_layers == 2 else 0) )
		self.num_layers = num_layers
		self.hidden_size = hidden_size
		if bidirectional: 
			self.num_direction = 2
		else: 
			self.num_direction = 1

		self.device = device

	def forward(self, input, lengths):
		bsize = input.shape[0]

		input, sorted_lengths, unsorted_idx = self.sort_sequences(input, lengths)
		sorted_lengths = sorted_lengths.cpu()
		packed = torch.nn.utils.rnn.pack_padded_sequence(input, sorted_lengths, batch_first = True)
		outputs, (h_n, c_n) = self.rnn(packed) # output: (seq_len, batch, hidden*n_dir)

		# print(outputs.shape)
		c_n = c_n.index_select(1, unsorted_idx.to(self.device))

		# h0 of shape : (num_layers * num_directions, batch, hidden_size)		
		# (num_layers, num_directions, batch, hidden_size)
		c_n = c_n.view(self.num_layers, self.num_direction, bsize, self.hidden_size)
		c_n = c_n.permute(0,2,1,3)
		c_n = c_n.reshape(self.num_layers, bsize, self.num_direction*self.hidden_size)

		if self.num_layers == 2:
			return c_n[1]
		if self.num_layers == 1:
			return c_n[0]

	def sort_sequences(self, inputs, lengths):
		"""sort_sequences
		Sort sequences according to lengths descendingly.

		:param inputs (Tensor): input sequences, size [B, T, D]
		:param lengths (Tensor): length of each sequence, size [B]
		"""
		lengths_sorted, sorted_idx = lengths.sort(descending=True)
		_, unsorted_idx = sorted_idx.sort()
		return inputs[sorted_idx], lengths_sorted, unsorted_idx

class label_cl(nn.Module):
	def __init__(self, num_classes, hidden_size, drop_rate):
		super(label_cl, self).__init__()
		self.hidden1 = nn.Linear(hidden_size, num_classes)
		# self.hidden2 = nn.Linear(hidden_size // 2, hidden_size // 4)
		# self.hidden3 = nn.Linear(hidden_size // 4, num_classes)
		# self.batch = nn.BatchNorm1d(hidden_size // 2)
		self.dropout1 = nn.Dropout(drop_rate)
		# self.dropout2 = nn.Dropout(0.25)

	def forward(self, x):
		# x = F.relu(self.dropout1(self.hidden1(x)))
		# x = F.relu(self.dropout2(self.hidden2(x)))
		x = self.dropout1(self.hidden1(x))

		return x


#  rnn = nn.LSTM(10, 20, 2)
# >>> input = torch.randn(5, 3, 10)
# >>> h0 = torch.randn(2, 3, 20)
# >>> c0 = torch.randn(2, 3, 20)
# >>> output, (hn, cn) = rnn(input, (h0, c0))