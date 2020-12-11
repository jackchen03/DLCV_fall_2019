import pandas as pd
import numpy as np
import argparse

def arg_parse():
	parser = argparse.ArgumentParser(description='domain generalization for the vlcs dataset')

	parser.add_argument('--pred_dir', type=str, default='OP01-R02-TurkeySandwich.txt', 
					help="root path to data directory")
	parser.add_argument('--label_dir', type=str, default='hw4_data/FullLengthVideos/labels/valid/OP01-R02-TurkeySandwich.txt', 
					help="root path to data directory")

	args = parser.parse_args()

	return args

if __name__=='__main__':
	args = arg_parse()

	label_arr = np.genfromtxt(args.label_dir, delimiter='\n')
	pred_arr = np.genfromtxt(args.pred_dir, delimiter='\n')

	acc_num = 0
	for i in range(pred_arr.shape[0]):
		if pred_arr[i] == label_arr[i]:
			acc_num += 1 

	print(str(acc_num) + '/' + str(pred_arr.shape[0]) + '(' + str(acc_num/pred_arr.shape[0]) +')' )

