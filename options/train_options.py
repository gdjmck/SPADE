"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # for displays
        parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')
        parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')

        # for training
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay')
        parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--optimizer', type=str, default='adam')
        parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--no_TTUR', action='store_true', help='Use TTUR training scheme')
        parser.add_argument('--color_raster', action='store_true', help='Use color raster loss')
        parser.add_argument('--total_variation_loss', action='store_true', help='Use total variation loss')
        parser.add_argument('--L1_loss', action='store_true', help='Use L1 loss for generated image and real image')
        parser.add_argument('--pool_size', type=int, default=0, help='history pool for VAE input')
        parser.add_argument('--gram_matrix_loss', action='store_true', help='compute MSE loss between gram matrix')
        parser.add_argument('--lambda_gram', type=float, default=100000.0)
        parser.add_argument('--lambda_l1', type=float, default=1.0)
        parser.add_argument('--dont_see_real', action='store_true', help='generator doesnt see real image')
        parser.add_argument('--regressor', type=str, default='', help='checkpoint file for condition regressor')
        parser.add_argument('--cover_rate', type=float, default=0, help='percentage of the image to be covered training masked GauGAN')
        parser.add_argument('--lr_discount', type=float, default=0.01, help='learning rate discount factor for condition encoder')

        # the default values for beta1 and beta2 differ by TTUR option
        opt, _ = parser.parse_known_args()
        if opt.no_TTUR:
            parser.set_defaults(beta1=0.5, beta2=0.999)

        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--D_steps_per_G', type=int, default=1, help='number of discriminator iterations per generator iterations.')

        # for discriminators
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        parser.add_argument('--lambda_vgg', type=float, default=10.0, help='weight for vgg loss')
        parser.add_argument('--lambda_color', type=float, default=5.0, help='weight for color loss')
        parser.add_argument('--lambda_TV', type=float, default=10.0, help='weight for color loss')
        parser.add_argument('--lambda_attr', type=float, default=1.0, help='weight for attribute recon loss')
        parser.add_argument('--lambda_reg', type=float, default=1.0, help='weight for discriminator regularization')
        parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')
        parser.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')
        parser.add_argument('--netD', type=str, default='multiscale', help='(n_layers|multiscale|image)')
        parser.add_argument('--lambda_kld', type=float, default=0.05)
        parser.add_argument('--diff_aug', action='store_true', help='to activate differentiable augmentation')
        parser.add_argument('--regularize_D', action='store_true', help='apply gradient descent on discriminator output to real image')
        parser.add_argument('--variance', action='store_true', help='attach variance map of the image to discriminator input')
        self.isTrain = True
        return parser
