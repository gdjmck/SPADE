import json
import torch
import numpy as np
from osgeo import gdal
from data.base_dataset import BaseDataset

class RoadDataset(BaseDataset):
    def __init__(self):
        super(RoadDataset, self).__init__()
        self.opt = None
        self.label_data = None  # 主干道栅格数据
        self.target_data = None  # 待生成道路栅格数据
        self.sample_position = None  # 坐标数据
        self.data_index_list = []  # 坐标索引
        self.width = 0  # 采样宽度
        self.height = 0  # 采样高度
        self.dataset_size = 0  # 数据集大小

    @staticmethod
    def modify_commandline_options(parser, is_train):
        load_size = 256
        parser.set_defaults(load_size=load_size)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=13)
        parser.set_defaults(contain_dontcare_label=False)

        parser.add_argument('--raster_input', type=str, default='', help='raster file of main road data')
        parser.add_argument('--raster_output', type=str, default='', help='raster file of small road data')
        parser.add_argument('--sample_position', type=str, default='', help='available points to sample in raster file')
        parser.add_argument('--index_file', type=str, default='', help='index file corresponding to sample_position file')

        return parser

    def initialize(self, opt):
        self.label_data = gdal.Open(opt.raster_input).ReadAsArray()  # 2d matrix
        self.target_data = gdal.Open(opt.raster_output).ReadAsArray()  # 2d matrix
        self.sample_position = np.load(opt.sample_position)  # dict(xs, ys)
        with open(opt.index_file, 'r') as f:
            self.data_index_list = json.load(f)['{}_index'.format(opt.phase)]
        assert opt.crop_size % 2 == 0
        self.height = self.width = opt.crop_size
        self.dataset_size = min(opt.max_dataset_size, len(self.data_index_list))

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        index = self.data_index_list[index]
        x_pos = self.sample_position['xs'][index]
        y_pos = self.sample_position['ys'][index]
        patch_label = self.label_data[(y_pos-self.height//2): (y_pos+self.height//2),
                      (x_pos-self.width//2): (x_pos+self.width//2)]
        patch_target = self.target_data[(y_pos-self.height//2): (y_pos+self.height//2),
                    (x_pos-self.width//2): (x_pos+self.width//2)]
        assert patch_target.any()
        assert patch_label.any()
        return {'label': torch.tensor(patch_label[None, ...], dtype=torch.float32),
                'image': torch.tensor(patch_target[None, ...], dtype=torch.float32),
                'instance': torch.tensor(1)}