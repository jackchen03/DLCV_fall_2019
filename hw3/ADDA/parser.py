from __future__ import absolute_import
import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='image segmentation for dlcv hw2')

    # Datasets parameters
    parser.add_argument('--data_dir', type=str, default='../hw3_data/digits/', 
                    help="root path to data directory")
    parser.add_argument('--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")

    # Mode
    parser.add_argument('--tran_di', type = str, default = 'sv_mn', choices=['sv_mn', 'mn_sv'])
    
    # training parameters

    parser.add_argument('--epoch', default=50, type=int,
                    help="num of validation iterations")
    parser.add_argument('--val_epoch', default=1, type=int,
                    help="num of validation iterations")
    parser.add_argument('--batch_size', default=128, type=int,
                    help="train batch size")
    # parser.add_argument('--test_batch', default=32, type=int, 
    #                 help="test batch size")
    parser.add_argument('--lr', default=1e-4, type=float,
                    help="initial learning rate")
    parser.add_argument('--weight-decay', default=0.0005, type=float,
                    help="initial learning rate")

    # adv training
    parser.add_argument('--g_iter', default=1, type=int, 
                    help='In homework, please always set to 0')
    parser.add_argument('--d_iter', default=1, type=int, 
                    help='In homework, please always set to 0')
    parser.add_argument('--dis_step', default=200, type=int, 
                    help='In homework, please always set to 0')

    # resume trained model
    parser.add_argument('--resume', type=int, default=0, 
                    help="starting epoch")
    parser.add_argument('--resume_pth', type=str, default='source_mn_sv_new/', 
                    help="path to the trained D")
    parser.add_argument('--test_pth', type=str, default='adv_mn_sv_new/', 
                    help="path to the trained D")
    # parser.add_argument('--resume_s', type=str, default='source_mn_sv_new/model/Best_SCNN.pth', 
    #                 help="path to the trained D")
    # parser.add_argument('--resume_t', type=str, default='source_mn_sv_new/model/Best_TCNN.pth', 
    #                 help="path to the trained D")
    # parser.add_argument('--resume_d', type=str, default='source_mn_sv_new/model/Best_D_cl.pth', 
    #                 help="path to the trained D")
    # parser.add_argument('--resume_l', type=str, default='source_mn_sv_new/model/Best_L_cl.pth', 
    #                 help="path to the trained G")

    # others
    parser.add_argument('--save_dir', type=str, default='log/')
    parser.add_argument('--random_seed', type=int, default=999)

    # HW --hw_data_dir $1 --tar_domain $2 --out_path $3
    parser.add_argument('--hw_data_dir', type=str)
    parser.add_argument('--tar_domain', type=str)
    parser.add_argument('--out_path', type=str)

    args = parser.parse_args()

    return args
