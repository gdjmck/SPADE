import torch
import torch.nn as nn
import os
from models.networks.discriminator import NLayerRegressHeadDiscriminator
from options.train_options import TrainOptions
import util.util


class ConditionRegressor(NLayerRegressHeadDiscriminator):
    def __init__(self, opt, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(ConditionRegressor, self).__init__(opt, n_layers, norm_layer)
        self.tail = None

    def forward(self, input):
        feat = self.head(input)
        regress_out = self.regression(feat)
        batchSize = regress_out.size(0)
        regress_out = self.linear(regress_out.view(batchSize, -1))
        return regress_out

def load_network(net, label, epoch, opt, strict=True):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    save_path = os.path.join(save_dir, save_filename)
    weights = torch.load(save_path)
    net.load_state_dict(weights, strict=strict)
    return net



if __name__ == '__main__':
    # parse options
    opt = TrainOptions().parse()
    # model = ConditionRegressor(opt, norm_layer=nn.InstanceNorm2d)
    model = ConditionRegressor(opt)

    # load discriminator as regressor
    # should first try with strict=True to check the missing parameters
    model = util.util.load_network(model, 'regressor', 96, opt)



