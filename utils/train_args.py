import argparse
import os
"""
To do:
    大量的冗余命令行参数还没有完全除去，我先能够跑起来再说
"""
def train_argparse():
    parser = argparse.ArgumentParser(description='MPANet train process')
    parser.add_argument('--not_save',default=False,
                        help='if yes,onlu output terminal')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=400, type=int, metavar='N',
                        help='number of total epochs to run(default: 400)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', default=1, type=int,
                        metavar='N', help='batch size (default: 1)')
    parser.add_argument('--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate (default: 0.001)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                        metavar='W', help='weight decay (default: 1e-5)')
    parser.add_argument('--train_dataset', required=True, type=str)
    parser.add_argument('--val_dataset', type=str)
    parser.add_argument('--save_freq', type=int,default = 5)
    parser.add_argument('--ckpt_frequence', type=int, default=40)
    parser.add_argument('--modelname', default='MedT', type=str,
                        help='type of model')
    parser.add_argument('--cuda', default="on", type=str,
                        help='switch on/off cuda option (default: off)')
    parser.add_argument('--aug', default='off', type=str,
                        help='turn on img augmentation (default: False)')
    parser.add_argument('--load', default='default', type=str,
                        help='load a pretrained model')
    parser.add_argument('--save', default='default', type=str,
                        help='save the model')
    parser.add_argument('--direc', default='./medt', type=str,
                        help='directory to save')
    parser.add_argument('--crop', type=int, default=None)
    parser.add_argument('--imgsize', type=int, default=None)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--gray', default='no', type=str)
    parser.add_argument('--work-dir', default=os.getcwd(),
                            help='the work folder for storing results')
    parser.add_argument('--num_of_branch',default=2,type=int,help="Multi patch test,value = 1,global branch,value = 2,global branch local branch,value = 3 all the branches")
    parser.add_argument('--save_flag',default=None,help= 'save output_masks')
    parser.add_argument('--RESUME',default=False,type=bool,help = 'checkpoint')
    parser.add_argument('--checkpoint_path',type=str,default=None,help='save checkpoint')
    return parser
def test_argparser():
    parser = argparse.ArgumentParser(description='MPANet test process')
    parser.add_argument('--not_save', default=False,
                        help='if yes,only output terminal')
    parser.add_argument('--work-dir', default=os.getcwd(),
                        help='the work folder for storing results')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run(default: 1)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', default=1, type=int,
                        metavar='N', help='batch size (default: 8)')
    parser.add_argument('--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate (default: 0.01)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--test_dataset', type=str)
    parser.add_argument('--modelname', default='off', type=str,
                        help='name of the model to load')
    parser.add_argument('--cuda', default="on", type=str,
                        help='switch on/off cuda option (default: off)')
    parser.add_argument('--direc', default='./results', type=str,
                        help='directory to save')
    parser.add_argument('--crop', type=int, default=None)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--loaddirec', default='load', type=str, help='pth file_forma')
    parser.add_argument('--imgsize', type=int, default=None)
    parser.add_argument('--gray', default='no', type=str)
    parser.add_argument('--score_thesh', default=0.5, type=float, help='calculate AUC score ')
    parser.add_argument('--save_flag', default=False, type=bool, help='save output_masks')
    parser.add_argument('--csv_save', default=False, type=bool,
                        help='output save relative metrics result csv folder')
    parser.add_argument('--aug', default='off', type=str,
                        help='turn on img augmentation (default: False)')
    parser.add_argument('--csv_dir', default=os.getcwd(), type=str, help='directory to save csv')
    parser.add_argument('--csv_name', default=None, help='csv_name')
    parser.add_argument('--num_of_branch', default=2, type=int,
                        help="Multi patch test,value = 1,global branch,value = 2,global branch local branch,value = 3 all the branches")
    parser.add_argument('--roc', type=bool, default=False, help="roc on off")
    parser.add_argument('--roc_dir', default=os.getcwd(), help='directory to save csv')
    parser.add_argument('--roc_save', type=bool, default=False, help='flag = save _csv_format')