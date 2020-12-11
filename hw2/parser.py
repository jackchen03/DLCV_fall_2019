from __future__ import absolute_import
import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='image segmentation for dlcv hw2')

    # Datasets parameters
    parser.add_argument('--data_dir', type=str, default='data', 
                    help="root path to data directory")
    parser.add_argument('--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")

    # Model type
    parser.add_argument('--model_type', type = str, default = 'deeplabv3', choices=['unet', 'baseline', 'deeplab', 'deeplabv3'])

    # baseline model settings
    parser.add_argument('--norm_fn', type=str, default='none', 
                    help="norm layer for trans convolution")
    parser.add_argument('--acti_fn', type=str, default='relu', 
                    help="activation layer for trans convolution")

    # UNet model settings
    parser.add_argument('--bilinear', type = bool, default = False, help = 'Unet upsampling bilinear or not.')

    # deeplab model settings
    parser.add_argument('--output_stride', type = int, default = 16)
    
    # training parameters
    parser.add_argument('--gpu', default=0, type=int, 
                    help='In homework, please always set to 0')
    parser.add_argument('--epoch', default=100, type=int,
                    help="num of validation iterations")
    parser.add_argument('--val_epoch', default=1, type=int,
                    help="num of validation iterations")
    parser.add_argument('--train_batch', default=32, type=int,
                    help="train batch size")
    parser.add_argument('--test_batch', default=32, type=int, 
                    help="test batch size")
    parser.add_argument('--lr', default=0.0002, type=float,
                    help="initial learning rate")
    parser.add_argument('--weight-decay', default=0.0005, type=float,
                    help="initial learning rate")

    # data augmentation
    parser.add_argument('--vflip', default=True, type=bool)
    parser.add_argument('--color_jitter', default = False, type = bool)
    
    # resume trained model
    parser.add_argument('--resume', type=str, default='', 
                    help="path to the trained model")
    # others
    parser.add_argument('--save_dir', type=str, default='log')
    parser.add_argument('--random_seed', type=int, default=999)

    args = parser.parse_args()

    return args
