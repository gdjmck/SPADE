import torch
import torch.nn as nn
import functools
from models.pix2pix_model import Pix2PixModel
import models.networks as networks


class UNetModel(Pix2PixModel):
    def __init__(self, opt):
        super(UNetModel, self).__init__(opt)

        # set loss functions
        if opt.isTrain:
            self.criterion_attr = torch.nn.MSELoss()
            self.criterion_recon = torch.nn.L1Loss()
            self.criterionGAN = networks.GANLoss(opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)

    def preprocess_input(self, data):
        """
        生成条件图
        """
        # move to GPU and change data types
        data['label'] = data['label'].float()
        if self.use_gpu():
            data['label'] = data['label'].cuda()
            data['image'] = data['image'].cuda()
            data['condition'] = data['condition'].cuda()

        # # create one-hot label map
        # label_map = data['label']
        # bs, _, h, w = label_map.size()
        # nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
        #     else self.opt.label_nc
        # input_label = self.FloatTensor(bs, nc, h, w).zero_()
        # input_semantics = input_label.scatter_(1, label_map, 1.0)

        return data['label'], data['image'], data['condition']


    def discriminate(self, fake_image, real_image):
        fake_and_real = torch.cat([fake_image, real_image], 0)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        patch, condition = self.netD(fake_and_real)

        patch_fake, patch_real = self.divide_pred(patch)
        condition_fake, condition_real = self.divide_pred(condition)

        return patch_fake, condition_fake, patch_real, condition_real

    def generate_fake(self, input_semantics, condition):
        return self.netG(input_semantics, condition)

    def compute_generator_loss(self, input_semantics, real_image, condition):
        G_losses = {}

        fake_image = self.generate_fake(input_semantics, condition)
        patch_fake, condition_fake, _, _ = self.discriminate(fake_image, real_image)

        G_losses['GAN'] = self.criterionGAN(patch_fake, True, for_discriminator=False)
        G_losses['recon'] = self.opt.lambda_l1 * self.criterion_recon(fake_image, real_image)
        G_losses['attr'] = self.criterion_attr(condition_fake, condition)

        return G_losses, fake_image

    def compute_discriminator_loss(self, input_semantics, real_image, condition):
        D_losses = {}
        with torch.no_grad():
            fake_image = self.generate_fake(input_semantics, condition)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()  # 为什么discriminator需要对生成的图片back propagate

        patch_fake, _, patch_real, condition_real = self.discriminate(fake_image, real_image)

        D_losses['D_fake'] = self.criterionGAN(patch_fake, False, for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(patch_real, True, for_discriminator=True)
        D_losses['attr'] = self.criterion_attr(condition_real, condition)

        return D_losses

    def forward(self, data, mode):
        input_semantics, real_image, condition = self.preprocess_input(data)

        if mode == 'inference':
            with torch.no_grad():
                fake_image = self.generate_fake(input_semantics, condition)
            return fake_image
        elif mode == 'generator':
            g_loss, fake_image = self.compute_generator_loss(input_semantics, real_image, condition)
            return g_loss, fake_image
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(input_semantics, real_image, condition)
            return d_loss
        else:
            raise ValueError("|mode| is invalid")


if __name__ == '__main__':
    import netron
    import hiddenlayer as h
    import torch.onnx
    from torch.autograd import Variable
    from torchviz import make_dot

    model = UNetGenerator(input_nc=1, output_nc=1, num_downs=8)
    model_file = './UnetGenerator.pth'
    x = torch.randn(1, 1, 256, 256)

    # # 使用netron分析模型
    # torch.onnx.export(model, x, model_file)
    # netron.start(model_file)

    # # 使用torchviz分析模型
    # out = model(x)
    # g = make_dot(out)
    # g.render(filename='graphviz', view=False, format='pdf')

    # 使用hiddenlayer分析模型
    graph = h.build_graph(model, x)
    graph.save(path='./hiddenlayer.png', format='png')
