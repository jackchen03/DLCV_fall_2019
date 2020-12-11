import pandas as pd
import numpy as np
import argparse

def arg_parse():
	parser = argparse.ArgumentParser(description='domain generalization for the vlcs dataset')

	parser.add_argument('--pred_dir', type=str, default='p1_valid.txt', 
					help="root path to data directory")

	args = parser.parse_args()

	return args

if __name__=='__main__':
	args = arg_parse()
	
	label_dir = 'hw4_data/TrimmedVideos/label/gt_valid.csv'
	df = pd.read_csv(label_dir)
	label_list = df['Action_labels']

	pred_arr = np.genfromtxt(args.pred_dir, delimiter='\n')

	acc_num = 0
	for i in range(pred_arr.shape[0]):
		if pred_arr[i] == label_list[i]:
			acc_num += 1 

	print(str(acc_num) + '/' + str(pred_arr.shape[0]) + '(' + str(acc_num/pred_arr.shape[0]) +')' )

