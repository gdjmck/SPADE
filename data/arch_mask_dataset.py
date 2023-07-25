import json
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from data.arch_dataset import ArchDataset
from data.base_dataset import get_params, get_transform, convert_label_image
from PIL import Image

DEBUG = False

class ArchMaskDataset(ArchDataset):
    """
    基于ArchDataset增加随机减少建筑
    """
    def mask_building(self, file, cover_rate: float=0.3):
        """
        调用Condition类的解析接口
        :param file:
        :return: Image
        """
        img = cv2.imread(file)
        if len(img.shape) == 3:
            img = img[..., 0]
        if DEBUG:
            plt.imsave('ori_img.png', img)
        # if img.shape[0] != self.condition_history.STANDARD_SIZE:
        #     fx = fy = self.condition_history.STANDARD_SIZE / img.shape[0]
        #     img = cv2.resize(img, dsize=None, fx=fx, fy=fy)
        # 从图片中解析出每个建筑
        json_file = file.replace('img', 'parse')[:-3] + 'json'
        try:
            with open(json_file, 'r') as f:
                build_info = json.load(f)
        except:
            print('重新识别')
            img, build_info = self.condition_history.parse_image(img)
            with open(json_file, 'w') as f:
                json.dump(build_info, f)
        if DEBUG:
            plt.imsave('pro_img.png', img)
        total_building = sum([len(build_list) for build_list in build_info.values()])
        assert total_building > 0
        num_build_to_mask = max(1, math.floor(total_building * cover_rate))

        build_mask_list = []
        for floor, outloop_list in build_info.items():
            for outloop in outloop_list:
                mask = np.zeros_like(img)
                cv2.fillPoly(mask, [np.array(outloop)], color=(1,))
                build_mask_list.append(mask)
        mask_index_list = np.random.choice(range(len(build_mask_list)), num_build_to_mask, replace=False)
        mask_acc = np.zeros_like(img)
        for mask_index in mask_index_list:
            mask_acc = mask_acc + build_mask_list[mask_index]
        if DEBUG:
            plt.imsave('mask_out.png', mask_acc)
        # remove image content in mask area
        img[mask_acc > 0] = 255
        if DEBUG:
            plt.imsave('masked_img.png', img)
        img = np.repeat(img[..., np.newaxis], 3, axis=-1)
        return Image.fromarray(img)


    def __getitem__(self, index):
        # Label Image
        label_path = self.label_paths[index]
        label = Image.open(label_path).convert('L')
        label = Image.fromarray(convert_label_image(np.array(label)))
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        # label像素值为255的未知类别
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc

        # input image (real images)
        image_path = self.image_paths[index]
        assert self.paths_match(label_path, image_path), \
            "The label_path %s and image_path %s don't match." % \
            (label_path, image_path)
        image = Image.open(image_path)
        image = image.convert('RGB')

        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)

        image_masked = self.mask_building(image_path, self.opt.cover_rate)
        image_masked_tensor = transform_image(image_masked)

        if False and self.opt.classify_color:
            # 将颜色tensor转换为类别的one-hot tensor
            image_tensor = self.parse_label(image_tensor[0].cpu().numpy())
            image_tensor = image_tensor.permute(2, 0, 1)
        elif self.opt.output_nc != image_tensor.size()[0]:
            image_tensor = image_tensor[:self.opt.output_nc]
            image_tensor = image_tensor[:1]

        input_dict = {'label': label_tensor,
                      'masked_image': image_masked_tensor,
                      'image': image_tensor,
                      'path': image_path,
                      }

        if self.opt.condition_size:
            # 添加回归属性
            condition = self.condition_history.get(self.image_paths[index])

            input_dict['condition'] = torch.tensor(condition, dtype=torch.float32)

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict
