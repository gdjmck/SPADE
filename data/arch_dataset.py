import cv2
import numpy as np
import torch
import torch.nn.functional as F
from data.custom_dataset import CustomDataset
from data.image_folder import make_dataset

COLOR_MAP = {i: 200 - i * 20 for i in range(11)}

def open_op(img_mask):
    return cv2.dilate(cv2.erode(img_mask, np.ones((3, 3), dtype=np.uint8), 3), np.ones((3, 3), dtype=np.uint8), 3)


class ArchDataset(CustomDataset):
    @classmethod
    def label2image(cls, label: np.ndarray):
        label = label.squeeze()
        label_max = label.max(0)
        image = np.zeros(label.shape[-2:], dtype=np.uint8)
        image[label[0] == label_max] = 255
        for index in range(1, label.shape[0]):
            image[label[index] == label_max] = COLOR_MAP[index-1]
        return image


    def initialize(self, opt):
        super(ArchDataset, self).initialize(opt)
        self.COLOR_MAP = {i: 200 - i * 20 for i in range(11)}


    def parse_label(self, img: np.ndarray):
        color_label_mask = np.zeros_like(img, dtype=np.uint8)
        # -1 ~ 1 -> 0 ~ 255
        img = (img + 1) * 255 / 2

        for colorId, color_val in self.COLOR_MAP.items():
            color_mask = ((img > color_val - 10) & (img < color_val + 10)).astype(np.uint8)  # full_size
            color_label_mask[color_mask > 0] = 1 + colorId  # full_size

            """
            # 合并小区域
            num_conn, comp_mask = cv2.connectedComponents(color_mask, connectivity=4)
            for connId in range(num_conn):
                if not ((comp_mask == connId) & color_mask).any():
                    continue
                # 遍历每个label的连通区域，尝试合并到大的类别中
                mask_current = (comp_mask == connId).astype(np.uint8)  # full_size
                mask_nb = cv2.dilate(mask_current, (3, 3), 3) ^ mask_current  # full_size
                mask_nb = (color_label_mask != 0) & mask_nb
                color_nb = color_label_mask[mask_nb > 0]

                attraction = {}

                for nb_label in np.unique(color_nb[color_nb > 0]).tolist():
                    num_conn, comp_mask = cv2.connectedComponents((color_label_mask == nb_label).astype(np.uint8),
                                                                  connectivity=4)
                    for conn_id in range(num_conn):
                        component = comp_mask == conn_id
                        if (component & mask_nb).any():
                            mass_nb = component.sum()
                            attraction[nb_label] = mass_nb

                if attraction:
                    target_label = max(attraction.keys(), key=lambda k: attraction[k])
                    if attraction[target_label] / mask_current.sum() > 2:
                        color_label_mask[mask_current > 0] = target_label
                        # print('颜色{}归为{}，吃掉{}'.format(1 + colorId, target_label, mask_current.sum()))
            """

        # convert to pixelwise one-hot tensor
        label_tensor = F.one_hot(torch.LongTensor(color_label_mask))
        return label_tensor


    def get_paths(self, opt):
        """
        opt.label_dir 为label文件夹路径列表字符串，以;为分隔符
        opt.image_dir 为image文件夹路径列表字符串，以;为分隔符
        """
        # gather label files
        label_dir_list = opt.label_dir.split(';')
        label_paths = []
        for label_dir in label_dir_list:
            label_paths.extend(make_dataset(label_dir, recursive=False, read_cache=True))

        # gather image files
        image_dir_list = opt.image_dir.split(';')
        image_paths = []
        for image_dir in image_dir_list:
            image_paths.extend(make_dataset(image_dir, recursive=False, read_cache=True))

        # gather instance files
        instance_paths = []
        if len(opt.instance_dir):
            instance_dir_list = opt.instance_dir.split(';')
            for instance_dir in instance_dir_list:
                instance_paths.extend(make_dataset(instance_dir, recursive=False, read_cache=True))

        assert len(label_paths) == len(image_paths), "The #images in %s and %s do not match. Is there something wrong?"

        return label_paths, image_paths, instance_paths
    