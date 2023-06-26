"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
from models.networks.sync_batchnorm import DataParallelWithCallback
from models.pix2pix_model import Pix2PixModel
from models import create_model
from torch.utils.tensorboard import SummaryWriter


class Pix2PixTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt):
        self.opt = opt
        # self.pix2pix_model = Pix2PixModel(opt)
        self.pix2pix_model = create_model(opt)
        if len(opt.gpu_ids) > 0:
            self.pix2pix_model = DataParallelWithCallback(self.pix2pix_model,
                                                          device_ids=opt.gpu_ids)
            self.pix2pix_model_on_one_gpu = self.pix2pix_model.module
        else:
            self.pix2pix_model_on_one_gpu = self.pix2pix_model

        self.generated = None
        if opt.isTrain:
            self.optimizer_G, self.optimizer_D = \
                self.pix2pix_model_on_one_gpu.create_optimizers(opt)
            self.old_lr = opt.lr

        # create summary writer
        self.summary_writer = SummaryWriter(log_dir=os.path.join(opt.checkpoints_dir, opt.name), comment='UNet')

    def log_histogram(self, step_index, model_type='D'):
        if model_type == 'D':
            try:
                param_iterator = self.pix2pix_model.netD.named_parameters()
            except AttributeError:
                param_iterator = self.pix2pix_model.module.netD.named_parameters()
            for name, param in param_iterator:
                try:
                    self.summary_writer.add_histogram(f'{name}.grad', param.grad, step_index)
                except:
                    # print('{} got no param'.format(name))
                    pass
        else:
            try:
                param_iterator = self.pix2pix_model.netG.named_parameters()
            except AttributeError:
                param_iterator = self.pix2pix_model.module.netG.named_parameters()
            for name, param in param_iterator:
                try:
                    self.summary_writer.add_histogram(f'{name}.grad', param.grad, step_index)
                except:
                    # print('{} got no param'.format(name))
                    pass

    def log_loss(self, loss_dict: dict, step_index: int, phase: str):
        """
        tensorboard记录损失值
        :param loss_dict: 损失字典
        :param step_index: 全局步数
        :param phase: 训练阶段，[G, D, E]
        :return:
        """
        self.summary_writer.add_scalars(main_tag=phase, tag_scalar_dict=loss_dict, global_step=step_index)

    def run_generator_one_step(self, data):
        self.optimizer_G.zero_grad()
        g_losses, generated = self.pix2pix_model(data, mode='generator')
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()
        self.optimizer_G.step()
        self.g_losses = g_losses
        self.generated = generated

    def run_discriminator_one_step(self, data):
        self.optimizer_D.zero_grad()
        d_losses = self.pix2pix_model(data, mode='discriminator')
        d_loss = sum(d_losses.values()).mean()
        d_loss.backward()
        self.optimizer_D.step()
        self.d_losses = d_losses

    def get_latest_losses(self):
        return {**self.g_losses, **self.d_losses}

    def get_latest_generated(self):
        return self.generated

    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    def save(self, epoch):
        self.pix2pix_model_on_one_gpu.save(epoch)

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr
