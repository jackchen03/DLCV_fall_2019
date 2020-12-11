from __future__ import absolute_import
import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='image segmentation for dlcv hw2')

    # Datasets parameters
    parser.add_argument('--data_dir', type=str, default='../hw3_data/face/train/', 
                    help="root path to data directory")
    parser.add_argument('--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")

    # Model type
    parser.add_argument('--model_type', type = str, default = 'baseline', choices=['baseline'])
    
    # training parameters

    parser.add_argument('--epoch', default=100, type=int,
                    help="num of validation iterations")
    parser.add_argument('--val_epoch', default=1, type=int,
                    help="num of validation iterations")
    parser.add_argument('--batch_size', default=64, type=int,
                    help="train batch size")
    # parser.add_argument('--test_batch', default=32, type=int, 
    #                 help="test batch size")
    parser.add_argument('--lr', default=0.0002, type=float,
                    help="initial learning rate")
    parser.add_argument('--weight-decay', default=0.0005, type=float,
                    help="initial learning rate")

    # GAN param
    parser.add_argument('--g_iter', type=int, default=2)

    # GAN type
    parser.add_argument('--gan_type', type = str, default = 'normal', choices=['normal', 'wgan', 'wgan-gp', 'lsgan'])

    # WGAN-GP param
    parser.add_argument('--lamda_gp', type=float, default=1.0)

    # resume trained model
    parser.add_argument('--load_dir', type=str, default='log/', 
                    help="path to the trained D")
    parser.add_argument('--load_epoch', type=int, default=80, 
                    help="path to the trained G")

    # others
    parser.add_argument('--save_dir', type=str, default='log/')
    parser.add_argument('--random_seed', type=int, default=999)

    # hw 
    parser.add_argument('--out_path', type=str)

    args = parser.parse_args()

    return args
