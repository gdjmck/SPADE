"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import sys
from collections import OrderedDict

import torch
import data
from options.test_options import TestOptions
from options.train_options import TrainOptions
from options.validate_options import ValidateOptions
from models.pix2pix_model import Pix2PixModel
from models import create_model, plot_model
from util.visualizer import Visualizer
from util import html
from util.volume_rate import Condition
from torch.utils.tensorboard import SummaryWriter
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

try:
    opt = TestOptions().parse()
except:
    opt = ValidateOptions().parse()
    # opt.isTrain = True
    opt.isValidate = True
    opt.fix_condition = False

dataloader = data.create_dataloader(opt)

# model = Pix2PixModel(opt)
model = create_model(opt)
model.eval()

if opt.just_plot:
    channels = opt.input_nc
    if opt.contain_dontcare_label:
        channels += 1
    input_size = (1, channels, opt.crop_size, opt.crop_size)
    plot_model(model.netG, input_size)
    sys.exit(0)

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

summary_writer = SummaryWriter(log_dir=web_dir, comment='Test')

# test
condition_on = opt.model != 'pix2pix'
if condition_on:
    pre_condition = Condition(opt)
for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break

    if condition_on:
        if opt.fix_condition:
            # 尝试固定条件
            data_i['condition'] = torch.Tensor([pre_condition.get(
                r'd:\Documents\aisr\GeosRelate\dataset_style3_slim\ArrangeMode\ColumnRow\arch_GZ\27.jpg').tolist()]).to(
                data_i['label'].device)
        generated, condition_fake, condition_real = model(data_i, mode='inference')
    else:
        generated = model(data_i, mode='inference')

    img_path = data_i['path']
    for b in range(generated.shape[0]):
        print('process image... %s' % img_path[b])
        visuals = OrderedDict([('input_label', data_i.get('image_masked', data_i['label'])[b]),
                               ('synthesized_image', generated[b])])
        visualizer.save_images(webpage, visuals, img_path[b:b + 1])
        ssim_val = ssim((generated + 1) / 2, (data_i['image'] + 1) / 2, data_range=1, size_average=True)
        print("SSIM: {}".format(ssim_val))
        if condition_on:
            gen_image_file = os.path.join(webpage.get_image_dir(), 'synthesized_image',
                                          img_path[b].rsplit('\\', 1)[1].replace('jpg', 'png'))
            assert os.path.exists(gen_image_file)
            condition_of_fake = pre_condition.cal_condition(gen_image_file, False)
            condition_real = pre_condition.read_condition(condition_real.cpu().numpy())
            condition_fake = pre_condition.read_condition(condition_fake.cpu().numpy())
            condition_read = pre_condition.read_condition(data_i['condition'].cpu().numpy())
            print('真实的条件:{}\n预测真实图片的条件:{}\n预测生成图片的条件:{}\n生成图片的真实条件:{}'.format(
                condition_read, condition_real, condition_fake, condition_of_fake))
            for tag, index in zip(pre_condition.condition_name, range(len(pre_condition.condition_name))):
                summary_writer.add_scalars(main_tag='Condition.{}'.format(tag),
                                           tag_scalar_dict={'real_read': condition_read[0][index],
                                                            'real_pred': condition_real[0][index],
                                                            'fake_read': condition_of_fake[index],
                                                            'fake_pred': condition_fake[0][index],
                                                            },
                                           global_step=i)
                summary_writer.add_scalar(tag='SSIM', scalar_value=ssim_val,
                                          global_step=i)

webpage.save()
summary_writer.close()
