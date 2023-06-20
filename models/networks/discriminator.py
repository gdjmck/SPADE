"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
import torch.nn.utils.spectral_norm as spectral_norm
import util.util as util
import functools
import math


class MultiscaleDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--netD_subarch', type=str, default='n_layer',
                            help='architecture of each discriminator')
        parser.add_argument('--num_D', type=int, default=2,
                            help='number of discriminators to be used in multiscale')
        opt, _ = parser.parse_known_args()

        # define properties of each discriminator of the multiscale discriminator
        subnetD = util.find_class_in_module(opt.netD_subarch + 'discriminator',
                                            'models.networks.discriminator')
        subnetD.modify_commandline_options(parser, is_train)

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        for i in range(opt.num_D):
            subnetD = self.create_single_discriminator(opt)
            self.add_module('discriminator_%d' % i, subnetD)

    def create_single_discriminator(self, opt):
        subarch = opt.netD_subarch
        if subarch == 'n_layer':
            netD = NLayerDiscriminator(opt)
        else:
            raise ValueError('unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input):
        result = []
        get_intermediate_features = not self.opt.no_ganFeat_loss
        for name, D in self.named_children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)

        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--n_layers_D', type=int, default=4,
                            help='# layers in each discriminator')
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = opt.ndf
        input_nc = self.compute_D_input_nc(opt)

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)
        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, opt.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == opt.n_layers_D - 1 else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                               stride=stride, padding=padw)),
                          nn.LeakyReLU(0.2, False)
                          ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def compute_D_input_nc(self, opt):
        input_nc = opt.label_nc + 1  # opt.output_nc
        if opt.contain_dontcare_label:
            input_nc += 1
        if not opt.no_instance:
            input_nc += 1
        return input_nc

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = not self.opt.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]


class UNetDiscriminator(BaseNetwork):
    """Defines a PatchGAN discriminator"""

    def __init__(self, opt, n_layers=2, norm_layer=nn.BatchNorm2d):
        """
        n_layers：(包含第一步下采样卷积)总卷积次数
        opt.ndf: default: 32
        """
        super(UNetDiscriminator, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            self.use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            self.use_bias = norm_layer == nn.InstanceNorm2d

        self.kw = 4
        self.padw = 1
        self.condition_size = opt.condition_size  # 回归值的个数
        norm_layer = get_nonspade_norm_layer(self.opt, self.opt.norm_D)
        # self.norm_layer = norm_layer

        # 共同特征提取头
        head_layers = [nn.Conv2d(opt.input_nc, opt.ndf, kernel_size=self.kw, stride=2, padding=self.padw),
                       nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            head_layers += [self.conv_block(opt.ndf * nf_mult_prev, opt.ndf * nf_mult, stride=2, norm_layer=norm_layer)]

        if 'spectral' in self.opt.norm_D:
            head_layers = [self._apply_spectral_norm(layer) for layer in head_layers]
        self.head = nn.Sequential(*head_layers)

        # Patch Discriminator
        patch_layers = []
        # channel 2*ndf -> 4*ndf, keeps spatial dimension
        # channel 4*ndf -> 8*ndf, keeps spatial dimension
        patch_layers += [
            self.conv_block(2 * opt.ndf, 4 * opt.ndf, kernel_size=3, stride=1, padding=1, norm_layer=norm_layer),
            self.conv_block(4 * opt.ndf, 8 * opt.ndf, kernel_size=3, stride=1, padding=1, norm_layer=norm_layer)]
        patch_layers += [nn.Conv2d(8 * opt.ndf, 1, kernel_size=3, stride=1, padding=self.padw, bias=self.use_bias)]

        if 'spectral' in self.opt.norm_D:
            patch_layers = [self._apply_spectral_norm(layer) for layer in patch_layers]
        self.patch_discriminator = nn.Sequential(*patch_layers)

        # Regression Head
        regression_layers = [self.conv_block(2 * opt.ndf, 4 * opt.ndf, stride=2),
                             self.conv_block(4 * opt.ndf, 4 * opt.ndf, stride=2),
                             self.conv_block(4 * opt.ndf, 8 * opt.ndf, stride=2)]
        regression_layers += [nn.Conv2d(8 * opt.ndf, 8 * opt.ndf, kernel_size=self.kw, stride=2, padding=self.padw),
                              nn.ReLU(),
                              nn.Conv2d(8 * opt.ndf, self.condition_size, kernel_size=self.kw, stride=1, padding=0)]

        if 'spectral' in self.opt.norm_D:
            regression_layers = [self._apply_spectral_norm(layer) for layer in regression_layers]
        self.regression = nn.Sequential(*regression_layers)

    def conv_block(self, in_nc: int, out_nc: int, stride: int, padding=None, kernel_size=None,
                   norm_layer=nn.BatchNorm2d):
        padding = padding if padding else self.padw
        kernel_size = kernel_size if kernel_size else self.kw
        conv = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride,
                         bias=self.use_bias, padding=padding)
        # if 'spectral' in self.opt.norm_D:
        #     conv = self._apply_spectral_norm(conv)
        try:
            conv = norm_layer(conv)
        except:
            pass
        return nn.Sequential(conv, nn.LeakyReLU(0.2))

    def forward(self, input):
        """Standard forward."""
        b, c, h, w = input.size()
        base_feature = self.head(input)
        discrimination = self.patch_discriminator(base_feature)
        if self.opt.condition_size:
            condition_regression = self.regression(base_feature)
            condition_regression = condition_regression.view(b, self.condition_size)
        else:
            condition_regression = None
        return discrimination, condition_regression


class NLayerRegressHeadDiscriminator(BaseNetwork):
    def __init__(self, opt, n_layers=3, norm_layer=nn.InstanceNorm2d):
        super(NLayerRegressHeadDiscriminator, self).__init__()
        self.opt = opt
        self.use_sigmoid = False
        self.n_layers = n_layers

        self.kw = 4
        self.padw = int(np.ceil((self.kw - 1.0) / 2))
        input_nc = opt.input_nc + opt.label_nc
        if opt.contain_dontcare_label:
            input_nc += 1
        sequence = [[nn.Conv2d(input_nc, opt.ndf, kernel_size=self.kw, stride=2, padding=self.padw), nn.LeakyReLU(0.2, True)]]

        ndf = opt.ndf
        nf = opt.ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=self.kw, stride=2, padding=self.padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=self.kw, stride=1, padding=self.padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence_stream = []
        for n in range(len(sequence)):
            sequence_stream += sequence[n]
        self.head = nn.Sequential(*sequence_stream)
        self.tail = nn.Conv2d(nf, 1, kernel_size=self.kw, stride=1, padding=self.padw)

        # 回归分支
        self.regression = []
        for i in range(2):
            self.regression += [nn.Conv2d(math.ceil(8 * ndf / 2 ** i), math.ceil(4 * ndf / 2 ** i),
                                          kernel_size=self.kw, stride=2, padding=self.padw),
                                nn.LeakyReLU(0.2, True)]
        self.regression += [nn.AdaptiveAvgPool2d(1)]
        self.regression = nn.Sequential(*self.regression)
        self.linear = nn.Linear(2 * ndf, opt.condition_size)

    def forward(self, input):
        """
        Return:
            disc_out: 鉴别器输出
            regress_out：回归头输出
        """
        feat = self.head(input)
        disc_out = self.tail(feat)
        regress_out = self.regression(feat)
        batchSize = regress_out.size(0)
        regress_out = self.linear(regress_out.view(batchSize, -1))
        return disc_out, regress_out


if __name__ == '__main__':
    import hiddenlayer as h
    import torch.onnx
    from options.train_options import TrainOptions

    # parse options
    opt = TrainOptions().parse()

    model = UNetDiscriminator(opt).regression
    model_file = './UNetDiscriminator.pth'
    x = torch.randn(1, 64, 64, 64)

    # 使用hiddenlayer分析模型
    graph = h.build_graph(model, x)
    graph.save(path='./hiddenlayer.png', format='png')
