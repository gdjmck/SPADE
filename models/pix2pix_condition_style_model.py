from models.pix2pix_condition_model import Pix2PixConditionModel
from models.networks.discriminator import NLayerRegressHeadDiscriminator
import util.util as util
import torch.nn as nn
import torch


class Pix2PixConditionStyleModel(Pix2PixConditionModel):

    def initialize_networks(self, opt):
        super(Pix2PixConditionStyleModel, self).initialize_networks(opt)
        # blender for code & condition
        self.blender = nn.Linear(256 + opt.condition_size, 256)
        if opt.use_vae and (not opt.isTrain or opt.continue_train or (hasattr(opt, 'isValidate') and opt.isValidate)):
            self.blender = util.load_network(self.blender, 'blender', opt.which_epoch, opt)
        # Using pretrained regressor
        self.condition_regressor = NLayerRegressHeadDiscriminator(opt)
        regressor_weights = torch.load(opt.regressor)
        self.condition_regressor.load_state_dict(regressor_weights)
        self.condition_regressor.eval()

    def predict_condition(self, input_semantics, image):
        """
        使用预训练的条件回归模型预测条件
        :param input_semantics:
        :param image:
        :return:
        """
        input_disc = torch.cat([input_semantics, image], dim=1)
        _, condition_pred = self.condition_regressor(input_disc)
        return condition_pred

    def discriminate(self, condition_real, condition_fake, fake_image, real_image):
        disc_fake = self.netD(fake_image, condition_fake)
        disc_real = self.netD(real_image, condition_real)
        return disc_real, disc_fake

    def compute_discriminator_loss(self, input_semantics, real_image, condition):
        D_losses = {}
        with torch.no_grad():
            fake_image, z_real = self.generate_fake(input_semantics, real_image, condition)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        if self.opt.regularize_D:
            real_image.requires_grad = True

        # generate z_fake for fake condition
        condition_fake = self.predict_condition(input_semantics, fake_image)

        z_fake = torch.randn(input_semantics.size(0), self.opt.z_dim,
                             dtype=torch.float32, device=input_semantics.get_device())
        z_fake = torch.cat([z_fake, condition_fake], 1)
        z_fake = self.blender(z_fake)

        patch_real, patch_fake = self.discriminate(z_real, z_fake, fake_image, real_image)

        # pred_fake down to -1
        D_losses['D_Fake'] = self.criterionGAN(patch_fake, False,
                                               for_discriminator=True)
        # pred_real up to 1
        D_losses['D_real'] = self.criterionGAN(patch_real, True,
                                               for_discriminator=True)
        return D_losses

    def generate_fake(self, input_semantics, real_image, condition=None):
        assert self.opt.use_vae
        # 条件融合
        if real_image is not None:
            z, mu, logvar = self.encode_z(real_image)
        else:
            z = torch.randn(input_semantics.size(0), self.opt.z_dim,
                            dtype=torch.float32, device=input_semantics.get_device())
            mu, logvar = None, None
        if condition is not None:
            z = torch.cat([z, condition], 1)
            z = self.blender(z)
        if self.opt.gram_matrix_loss:
            fake_image, gram_matrix = self.netG(input_semantics, z=z, return_gram=True)
        else:
            fake_image = self.netG(input_semantics, z=z)
        return fake_image, z

    def compute_generator_loss(self, input_semantics, real_image, condition=None):
        G_losses = {}
        if self.opt.pool_size > 0:
            real_image_for_vae = self.image_pool.query(real_image)
            fake_image, condition_fake = self.generate_fake(
                input_semantics, real_image_for_vae, condition)
        elif self.opt.dont_see_real:
            fake_image, condition_fake = self.generate_fake(
                input_semantics, None, condition)
        else:
            fake_image, condition_fake = self.generate_fake(
                input_semantics, real_image, condition)

        _, patch_fake = self.discriminate(condition, condition_fake, fake_image, real_image)

        # maximize pred_fake
        G_losses['GAN'] = self.criterionGAN(patch_fake, True, for_discriminator=False)

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
                fake_image = torch.cat([fake_image] * 3, 1)
                real_image = torch.cat([real_image] * 3, 1)
            G_losses['VGG'] = self.criterionVGG(fake_image, real_image) * self.opt.lambda_vgg

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
