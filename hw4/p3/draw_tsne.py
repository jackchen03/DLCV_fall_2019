from tsne import tSNE
import parser
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__=='__main__':
	args = parser.arg_parse()

	tsne = tSNE(args, device)
	tsne.draw()