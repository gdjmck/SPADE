"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import math
from models.networks.encoder import LabelEncoder
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import ResnetBlock as ResnetBlock
from models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock


class SPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        # parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.set_defaults(norm_G='spectralspadeinstance3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        if opt.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2

        if self.opt.classify_color:
            self.conv_img = nn.Conv2d(final_nc, self.opt.output_nc, 1, padding=0)
        else:
            self.conv_img = nn.Conv2d(final_nc, self.opt.output_nc, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def forward(self, input, z=None, return_gram=False):
        seg = input

        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        x = self.head_0(x, seg)

        x = self.up(x)
        x = self.G_middle_0(x, seg)

        if self.opt.num_upsampling_layers == 'more' or \
           self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg)

        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg)

        # gram matrix adds here
        if return_gram:
            gram_matrix = self._compute_gram_matrix(x)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        if not self.opt.classify_color:
            x = F.tanh(x)

        if return_gram:
            return x, gram_matrix
        else:
            return x


class SPADEStyleGenerator(SPADEGenerator):
    def __init__(self, opt):
        """
        将SPADEGenerator的SPADEResnetBlock替换为ConditionalSPADEResnetBlock
        :param opt:
        """
        super(SPADEStyleGenerator, self).__init__(opt)
        from models.networks.architecture import ConditionalSPADEResnetBlock as CSPADEResnetBlock
        self.label_encoder = LabelEncoder(opt, 8 if opt.num_upsampling_layers == 'most' else 7)
        self.iter_index = 0
        nf = opt.ngf
        self.head_0 = CSPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = CSPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = CSPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = CSPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = CSPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = CSPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = CSPADEResnetBlock(2 * nf, 1 * nf, opt)

        if opt.num_upsampling_layers == 'most':
            self.up_4 = CSPADEResnetBlock(1 * nf, nf // 2, opt)

    def iter(self, feat):
        self.iter_index += 1
        return feat[:, self.iter_index-1]

    def forward(self, input, z, return_gram=False):
        """

        :param input: semantic map
        :param z: condition tensor
        :param return_gram:
        :return:
        """
        seg = input
        self.iter_index = 0

        z = self.label_encoder(z)

        noise = torch.randn(input.size(0), self.opt.z_dim,
                            dtype=torch.float32, device=input.get_device())
        x = self.fc(noise)
        x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)

        x = self.head_0(x, seg, self.iter(z))

        x = self.up(x)
        x = self.G_middle_0(x, seg, self.iter(z))

        if self.opt.num_upsampling_layers == 'more' or \
           self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg, self.iter(z))

        x = self.up(x)
        x = self.up_0(x, seg, self.iter(z))
        x = self.up(x)
        x = self.up_1(x, seg, self.iter(z))
        x = self.up(x)
        x = self.up_2(x, seg, self.iter(z))
        x = self.up(x)
        x = self.up_3(x, seg, self.iter(z))

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg, self.iter(z))

        # gram matrix adds here
        if return_gram:
            gram_matrix = self._compute_gram_matrix(x)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        if not self.opt.classify_color:
            x = F.tanh(x)

        if return_gram:
            return x, gram_matrix
        else:
            return x


class Pix2PixHDGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--resnet_n_downsample', type=int, default=4, help='number of downsampling layers in netG')
        parser.add_argument('--resnet_n_blocks', type=int, default=4, help='number of residual blocks in the global generator network')
        parser.add_argument('--resnet_kernel_size', type=int, default=3,
                            help='kernel size of the resnet block')
        parser.add_argument('--resnet_initial_kernel_size', type=int, default=3,
                            help='kernel size of the first convolution')
        parser.set_defaults(norm_G='instance')
        return parser

    def __init__(self, opt):
        super().__init__()
        input_nc = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)
        activation = nn.ReLU(True)

        model = []

        # initial conv
        model += [nn.ReflectionPad2d(opt.resnet_initial_kernel_size // 2),
                  norm_layer(nn.Conv2d(input_nc, opt.ngf,
                                       kernel_size=opt.resnet_initial_kernel_size,
                                       padding=0)),
                  activation]

        # downsample
        mult = 1
        for i in range(opt.resnet_n_downsample):
            model += [norm_layer(nn.Conv2d(opt.ngf * mult, opt.ngf * mult * 2,
                                           kernel_size=3, stride=2, padding=1)),
                      activation]
            mult *= 2

        # resnet blocks
        for i in range(opt.resnet_n_blocks):
            model += [ResnetBlock(opt.ngf * mult,
                                  norm_layer=norm_layer,
                                  activation=activation,
                                  kernel_size=opt.resnet_kernel_size)]

        # upsample
        for i in range(opt.resnet_n_downsample):
            nc_in = int(opt.ngf * mult)
            nc_out = int((opt.ngf * mult) / 2)
            model += [norm_layer(nn.ConvTranspose2d(nc_in, nc_out,
                                                    kernel_size=3, stride=2,
                                                    padding=1, output_padding=1)),
                      activation]
            mult = mult // 2

        # final output conv
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(nc_out, opt.output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, z=None):
        return self.model(input)


class UNetGenerator(BaseNetwork):
    def __init__(self, opt, norm_layer=nn.BatchNorm2d):
        """
        num_downs: 最小为 5
        """
        super(UNetGenerator, self).__init__()
        self.opt = opt
        self.num_downs = math.ceil(math.log2(opt.crop_size))
        self.ngf = opt.ngf  # default: 64
        self.condition_size = opt.condition_size  # default: 5
        self.down_seq = []
        self.up_seq = []  # 最后做reverse
        # norm_layer = get_nonspade_norm_layer(self.opt, self.opt.norm_G)
        # UNet最外层
        self.down_seq.append(self.downconv(opt.input_nc, self.ngf, 'outer', norm_layer=norm_layer))
        self.up_seq.append(nn.Sequential(
            self.upconv(2 * self.ngf, self.ngf, 'outer', norm_layer=norm_layer),
            nn.BatchNorm2d(self.ngf),
            nn.LeakyReLU(),
            nn.Conv2d(self.ngf, opt.output_nc, kernel_size=1, padding=0)
        ))
        # norm_layer = get_nonspade_norm_layer(self.opt, self.opt.norm_G)
        # UNet中间层（特征深度有变化）
        for i in range(3):
            # outer_nc = ngf; inner_nc = 2*ngf; input_nc = ngf
            channel_ratio = 2 ** i
            self.down_seq.append(self.downconv(self.ngf * channel_ratio, 2 * self.ngf * channel_ratio,
                                               'middle', norm_layer=norm_layer))
            self.up_seq.append(self.upconv(4 * self.ngf * channel_ratio, self.ngf * channel_ratio,
                                           'middle', norm_layer=norm_layer))
        # UNet中间层（特征深度不变）
        for i in range(max(0, self.num_downs - 5)):
            self.down_seq.append(self.downconv(8 * self.ngf, 8 * self.ngf, 'middle', norm_layer=norm_layer))
            self.up_seq.append(self.upconv(16 * self.ngf, 8 * self.ngf, 'middle', norm_layer=norm_layer))
        # UNet最里层
        self.down_seq.append(self.downconv(8 * self.ngf, 8 * self.ngf, 'inner', norm_layer=norm_layer))
        self.up_seq.append(self.upconv(8 * self.ngf, 8 * self.ngf, 'inner', norm_layer=norm_layer))

        # 属性fusion层
        self.fusion = nn.Sequential(nn.Linear(8 * self.ngf + opt.condition_size, 8 * self.ngf),
                                    nn.ReLU())

        self.up_seq = self.to_moduleList(self.up_seq, reverse=True)
        self.down_seq = self.to_moduleList(self.down_seq, reverse=False)

        self.post_act = nn.Tanh()

    def to_moduleList(self, module_list: list, reverse=False):
        if reverse:
            module_list = module_list[::-1]
        return nn.ModuleList(module_list)

    def forward(self, x, condition_vec):
        shortcuts = []
        # 下采样过程
        for layer_i in range(len(self.down_seq)):
            x = self.down_seq[layer_i](x)
            # print('下采样_{}的输出尺寸:{}'.format(layer_i, x.size()))
            shortcuts.append(x)
        # 加入条件向量
        if self.condition_size:
            b, c, h, w = x.size()
            x_with_condition = torch.cat([x.view(b, -1), condition_vec], 1)
            x = self.fusion(x_with_condition).view(b, c, h, w)

        # 上采样过程
        for layer_i in range(len(self.up_seq)):
            feature_shortcut = shortcuts.pop()
            if layer_i != 0:
                # print('上采样_{}: 拼接尺寸 shortcut: {}与 当前: {}'.format(layer_i,
                #                                          feature_shortcut.size(),
                #                                          x.size()))
                x = torch.cat([x, feature_shortcut], 1)
            x = self.up_seq[layer_i](x)
            # print('上采样_{}输出:{}'.format(layer_i, x.size()))
        # output activation
        x = self.post_act(x)
        return x

    def downconv(self, in_nc: int, out_nc: int, down_type: str, norm_layer=nn.BatchNorm2d):
        # use_bias on
        use_bias = True

        down = [nn.Conv2d(in_nc, out_nc, kernel_size=4, stride=2, padding=1, bias=use_bias),
                norm_layer(out_nc),
                nn.Conv2d(out_nc, out_nc, kernel_size=1, stride=1, padding=0),
                norm_layer(out_nc)]
        if down_type == 'inner':
            # down_seq = []
            # for i, down_op in enumerate(down):
            #     if i == 0:
            #         down_seq.extend([down_op, nn.LeakyReLU(0.2)])
            #     else:
            #         down_seq.extend([down_op, norm_layer(out_nc), nn.LeakyReLU(0.2)])
            # down = down_seq
            down = [nn.MaxPool2d(2), nn.LeakyReLU(0.2)]
        elif down_type == 'middle':
            down_seq = []
            for down_op in down:
                down_seq.extend([down_op, nn.LeakyReLU(0.2)])
            down = down_seq
        else:  # down_type == 'outer'
            down = [down[0], nn.LeakyReLU(0.1)]

        # if 'spectral' in self.opt.norm_G:
        #     down = [self._apply_spectral_norm(layer) for layer in down]

        return nn.Sequential(*down)

    def upconv(self, in_nc: int, out_nc: int, up_type: str, norm_layer=nn.BatchNorm2d):
        # use_bias on
        use_bias = True

        up = [nn.ConvTranspose2d(in_nc, out_nc, kernel_size=4, stride=2, padding=1, bias=use_bias),
              norm_layer(out_nc)]
        if up_type in ['inner', 'middle']:
            up = up + [nn.LeakyReLU(0.2)]
        else:  # up_type == 'outer'
            # 修改最外层输出卷积操作
            up = [nn.ConvTranspose2d(in_nc, in_nc//2, kernel_size=4, stride=2, padding=1, bias=use_bias),
                  norm_layer(in_nc//2),
                  nn.LeakyReLU(0.2),
                  nn.Conv2d(in_nc//2, out_nc, kernel_size=1, stride=1, padding=0, bias=False),
                  norm_layer(out_nc)]
            up = up + [nn.Tanh()]
        # # apply spectral_norm
        # if 'spectral' in self.opt.norm_G:
        #     up = [self._apply_spectral_norm(layer) for layer in up]

        return nn.Sequential(*up)

