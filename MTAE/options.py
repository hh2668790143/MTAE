import argparse
import os
import torch


class Options():
    """Options class

    Returns:
        [argparse]: argparse containing train and test options
    """

    def __init__(self):

        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


        self.parser.add_argument('--dataset', type=str, default='NO', help='dataset name')
        self.parser.add_argument('--batchsize', type=int, default=128, help='input batch size')
        self.parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
        self.parser.add_argument('--isize', type=int, default=166, help='input sequence size.')

        self.parser.add_argument('--device', type=str, default='gpu', help='Device: gpu | cpu')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')

        self.parser.add_argument('--nc', type=int, default=1, help='input sequence channels')
        self.parser.add_argument('--nz', type=int, default=8, help='size of the latent z vector')
        self.parser.add_argument('--ngf', type=int, default=32)
        self.parser.add_argument('--ndf', type=int, default=32)

        ##
        # Train
        self.parser.add_argument('--print_freq', type=int, default=50,
                                 help='frequency of showing training results on console')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        self.parser.add_argument('--w_adv', type=float, default=1, help='parameter')
        self.parser.add_argument('--folder', type=int, default=0, help='folder index 0-4')
        self.parser.add_argument('--n_aug', type=int, default=0, help='aug data times')
        self.parser.add_argument('--signal_length', type=list, default=[48], help='the length of wavelet signal')

        self.parser.add_argument('--normal_idx', type=int, default=0, help='the label index of normaly')
        self.parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
        self.parser.add_argument('--early_stop', type=int, default=100)

        # ganomaly
        self.parser.add_argument('--extralayers', type=int, default=0, help='Number of extra layers on gen and disc')
        self.parser.add_argument('--decoder_isize', type=int, default=32, help='input sequence size.')
        self.parser.add_argument('--resume', default='', help="path to checkpoints (to continue training)")
        self.parser.add_argument('--w_con', type=float, default=50, help='Reconstruction loss weight')
        self.parser.add_argument('--w_enc', type=float, default=1, help='Encoder loss weight.')
        self.parser.add_argument('--save_image_freq', type=int, default=100,
                                 help='frequency of saving real and fake images')
        self.parser.add_argument('--load_weights', action='store_true', help='Load the pretrained weights')
        self.parser.add_argument('--save_test_images', action='store_true', help='Save test images for demo.')
        self.parser.add_argument('--metric', type=str, default='roc', help='Evaluation metric.')
        self.parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')
        self.parser.add_argument('--display', action='store_true', help='Use visdom.')

        ### DeepSVDD###
        self.parser.add_argument('--dataset_name', default='UCR', help='run dataset')
        self.parser.add_argument('--net_name', default='CNN',
                                 choices=['CNN'],
                                 help='Model name.')
        self.parser.add_argument('--n_epochs', type=int, default=300, help='Number of epochs to train.')
        self.parser.add_argument('--data_path', default='.', help='Dataset path')
        self.parser.add_argument('--load_config', default=None, help='Config JSON-file path (default: None).')
        self.parser.add_argument('--load_model', default=None, help='Model file path (default: None).')
        self.parser.add_argument('--objective', default='one-class', choices=['one-class', 'soft-boundary'],
                                 help='Specify Deep SVDD objective ("one-class" or "soft-boundary").')
        self.parser.add_argument('--nu', type=float, default=0.1,
                                 help='Deep SVDD hyperparameter nu (must be 0 < nu <= 1).')

        self.parser.add_argument('--optimizer_name', choices=['adam'], default='adam',
                                 help='Name of the optimizer to use for Deep SVDD network training.')

        self.parser.add_argument('--lr_milestone', type=list, default=[0],
                                 help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
        # self.parser.add_argument('--batch_size', type=int, default=32, help='Batch size for mini-batch training.')
        self.parser.add_argument('--weight_decay', type=float, default=1e-6,
                                 help='Weight decay (L2 penalty) hyperparameter for Deep SVDD objective.')
        self.parser.add_argument('--pretrain', type=bool, default=False,
                                 help='Pretrain neural network parameters via autoencoder.')
        self.parser.add_argument('--ae_optimizer_name', choices=['adam'], default='adam',
                                 help='Name of the optimizer to use for autoencoder pretraining.')
        self.parser.add_argument('--ae_lr', type=float, default=0.0001,
                                 help='Initial learning rate for autoencoder pretraining. Default=0.001')
        self.parser.add_argument('--ae_n_epochs', type=int, default=1000, help='Number of epochs to train autoencoder.')
        self.parser.add_argument('--ae_lr_milestone', type=list, default=[0],
                                 help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
        self.parser.add_argument('--ae_batch_size', type=int, default=128,
                                 help='Batch size for mini-batch autoencoder training.')
        self.parser.add_argument('--ae_weight_decay', type=float, default=1e-6,
                                 help='Weight decay (L2 penalty) hyperparameter for autoencoder objective.')
        self.parser.add_argument('--n_jobs_dataloader', type=int, default=0,
                                 help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
        self.parser.add_argument('--normal_class', type=int, default=0,
                                 help='Specify the normal class of the dataset (all other classes are considered anomalous).')

        self.opt = None
    def parse(self):
        """ Parse Arguments.
        """

        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if self.opt.device == 'gpu':
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        return self.opt
