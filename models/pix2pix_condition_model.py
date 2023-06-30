import torch
import torch.nn as nn
from .pix2pix_model import Pix2PixModel
import util.util as util
from util.image_pool import ImagePool

class Pix2PixConditionModel(Pix2PixModel):
    def __init__(self, opt):
        super(Pix2PixConditionModel, self).__init__(opt)

        if self.use_gpu():
            self.blender.cuda()
        if opt.isTrain:
            self.criterion_attr = torch.nn.MSELoss()
            self.image_pool = ImagePool(opt.pool_size)

    def initialize_networks(self, opt):
        super(Pix2PixConditionModel, self).initialize_networks(opt)
        # blender for code & condition
        self.blender = nn.Linear(256+opt.condition_size, 256)
        if not opt.isTrain or opt.continue_train or (hasattr(opt, 'isValidate') and opt.isValidate):
            self.blender = util.load_network(self.blender, 'blender', opt.which_epoch, opt)

    def generate_fake(self, input_semantics, real_image, compute_kld_loss=False, condition=None):
        assert self.opt.use_vae
        z = None
        KLD_loss = None
        # 条件融合
        z, mu, logvar = self.encode_z(real_image)
        if condition is not None:
            z = torch.cat([z, condition], 1)
            z = self.blender(z)
            if compute_kld_loss:
                KLD_loss = self.KLDLoss(mu, logvar) * self.opt.lambda_kld
        fake_image = self.netG(input_semantics, z=z)
        return fake_image, KLD_loss

    def preprocess_input(self, data):
        data_processed = super(Pix2PixConditionModel, self).preprocess_input(data)
        condition = data.get('condition', None)
        if condition is not None:
            condition = condition.to(data_processed[0].device)
        return list(data_processed[:-1]) + [condition]

    def discriminate(self, input_semantics, fake_image, real_image):
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        patch, condition = self.netD(fake_and_real)
        patch_fake, patch_real = self.divide_pred(patch)

        if self.opt.condition_size:
            condition_fake, condition_real = self.divide_pred(condition)
        else:
            condition_fake, condition_real = None, None

        return patch_fake, condition_fake, patch_real, condition_real

    def compute_generator_loss(self, input_semantics, real_image, condition=None):
        G_losses = {}
        if self.opt.pool_size > 0:
            real_image_for_vae = self.image_pool.query(real_image)
            fake_image, KLD_loss = self.generate_fake(
                input_semantics, real_image_for_vae, self.opt.use_vae, condition)
        else:
            fake_image, KLD_loss = self.generate_fake(
                input_semantics, real_image, self.opt.use_vae, condition)

        if self.opt.use_vae:
            G_losses['KLD'] = KLD_loss

        patch_fake, condition_fake, _, _ = self.discriminate(input_semantics, fake_image, real_image)

        # maximize pred_fake
        G_losses['GAN'] = self.criterionGAN(patch_fake, True, for_discriminator=False)


        if self.opt.condition_size:
            G_losses['G_attr'] = self.opt.lambda_attr * self.criterion_attr(condition_fake, condition)

        if self.opt.L1_loss:
            fg_mask = input_semantics[:, :1]
            G_losses['L1'] = (self.L1(fake_image, real_image) * fg_mask).sum() / fg_mask.sum()
            # G_losses['L1'] = self.opt.lambda_l1 * self.L1(fake_image * fg_mask, real_image * fg_mask)

        if not self.opt.no_ganFeat_loss:
            num_D = len(patch_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(patch_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        patch_fake[i][j], patch_fake[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        if not self.opt.no_vgg_loss:
            if fake_image.size(1) == 1:
                fake_image = torch.cat([fake_image]*3, 1)
                real_image = torch.cat([real_image]*3, 1)
            G_losses['VGG'] = self.criterionVGG(fake_image, real_image) \
                              * self.opt.lambda_vgg

        if self.opt.total_variation_loss:
            G_losses['TV'] = self.criterionTV(fake_image)

        # 前景区域颜色区间化
        if self.opt.color_raster:
            fake_image_upper, fake_image_lower, fake_image_bias = util.COLOR.dicretize(fake_image)
            mask_foreground = (1 - input_semantics[:, -1:, ...])
            G_losses['color_raster'] = self.opt.lambda_color * ((fake_image_bias -
                                                                 self.criterionColor(fake_image, fake_image_upper) -
                                                                 self.criterionColor(fake_image, fake_image_lower)
                                                                 ) * mask_foreground).sum() / mask_foreground.sum()
            # try:
            #     assert G_losses['color_raster'] >= 0
            # except AssertionError:
            #     print(G_losses['color_raster'].item())

        return G_losses, fake_image

    def compute_discriminator_loss(self, input_semantics, real_image, condition):
        D_losses = {}
        with torch.no_grad():
            fake_image, _ = self.generate_fake(input_semantics, real_image, condition)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        patch_fake, _, patch_real, condition_real = self.discriminate(
            input_semantics, fake_image, real_image)

        # pred_fake down to -1
        D_losses['D_Fake'] = self.criterionGAN(patch_fake, False,
                                               for_discriminator=True)
        # pred_real up to 1
        D_losses['D_real'] = self.criterionGAN(patch_real, True,
                                               for_discriminator=True)
        if self.opt.condition_size:
            D_losses['D_attr'] = self.opt.lambda_attr * self.criterion_attr(condition_real, condition)

        return D_losses

    def forward(self, data, mode):
        input_semantics, real_image, condition = self.preprocess_input(data)  # vr = None if not opt.volume_rate

        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(
                input_semantics, real_image, condition)
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                input_semantics, real_image, condition)
            return d_loss
        elif mode == 'encode_only':
            z, mu, logvar = self.encode_z(real_image)
            return mu, logvar
        elif mode == 'inference':
            with torch.no_grad():
                fake_image, _ = self.generate_fake(input_semantics, real_image, condition=condition)
                _, condition_fake, _, condition_real = self.discriminate(input_semantics, fake_image, real_image)
            return fake_image, condition_fake, condition_real
        else:
            raise ValueError("|mode| is invalid")
        
    def save(self, epoch):
        super(Pix2PixConditionModel, self).save(epoch)
        # additional blender
        util.save_network(self.blender, 'blender', epoch, self.opt)
        