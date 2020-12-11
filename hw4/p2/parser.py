from __future__ import absolute_import
import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='domain generalization for the vlcs dataset')

    # Datasets parameters
    parser.add_argument('--data_dir', type=str, default='../hw4_data/TrimmedVideos/video/', 
                    help="root path to data directory")
    parser.add_argument('--label_dir', type=str, default='../hw4_data/TrimmedVideos/label/', 
                    help="root path to data directory")
    parser.add_argument('--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
    parser.add_argument('--video_len', default=180, type=int,
                    help="number of data loading workers (default: 4)")
    parser.add_argument('--feat_dir', type=str, default='csv_152', 
                    help="root path to data directory")

    # RNN parameters
    parser.add_argument('--num_layer', default=2, type=int,
                    help="num of rnn layer")
    parser.add_argument('--bidirectional', default=True, type=bool,
                    help="bidirectional rnn or not")
    parser.add_argument('--drop_cl', default=0.2, type=float,
                    help="num of rnn layer")
    parser.add_argument('--drop_rnn', default=0.3, type=float,
                    help="num of rnn layer")
    
    # training parameters
    parser.add_argument('--epoch', default=100, type=int,
                    help="num of validation iterations")
    parser.add_argument('--val_epoch', default=2, type=int,
                    help="num of validation iterations")
    parser.add_argument('--batch_size', default=8, type=int,
                    help="train batch size")
    # parser.add_argument('--test_batch', default=32, type=int, 
    #                 help="test batch size")
    parser.add_argument('--lr', default=0.001, type=float,
                    help="initial learning rate")
    parser.add_argument('--weight-decay', default=0.0005, type=float,
                    help="initial learning rate")

    # resume trained model
    parser.add_argument('--resume', type=int, default=0, 
                    help="starting epoch")
    parser.add_argument('--resume_pth', type=str, default='log/', 
                    help="path to the trained D")
    parser.add_argument('--resume_epoch', type=int, default=-1, 
                    help="path to the trained D")
    parser.add_argument('--test_pth', type=str, default='adv_mn_sv_new/', 
                    help="path to the trained D")

    # others
    parser.add_argument('--save_dir', type=str, default='log/')
    parser.add_argument('--random_seed', type=int, default=999)

    # for HW
    parser.add_argument('--hw_video_dir', type=str)
    parser.add_argument('--hw_label_dir', type=str)
    parser.add_argument('--hw_out_dir', type=str)


    args = parser.parse_args()

    return args
