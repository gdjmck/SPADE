import cv2
import math
import numpy as np
import os
import json


class Condition:
    def __init__(self, opt):
        self.condition_dict = {}
        self.condition_size = 5
        try:
            with open(opt.condition_norm, 'r', encoding='utf-8') as f:
                condition_norm = json.load(f)
            self.condition_mean = np.array(condition_norm['mean'])
            self.condition_stdvar = np.array(condition_norm['stdvar'])
        except:
            self.condition_mean = np.array([0] * self.condition_size)
            self.condition_stdvar = np.array([1] * self.condition_size)
        self.floor_choice = [1+i*3 for i in range(11)]  # 采样层数 [1, 4, 7, ..., 31]
        self.MAX_AREA = 90000
        self.COLOR_MAP = {i: 200 - i * 20 for i in range(11)}  # 层数与颜色的映射

    def update_mean_and_stdvar(self):
        """
        根据已记录在案的数据计算均值与标准差
        """
        data = np.empty((len(self.condition_dict), self.condition_size), dtype=np.float32)
        for i, condition in enumerate(self.condition_dict.values()):
            data[i] = np.array(condition) * self.condition_stdvar + self.condition_mean

        # print('旧均值:{}\n旧标准差为:{}'.format(self.condition_mean.tolist(),
        #                                       self.condition_stdvar.tolist()))
        mean_update = data.mean(0)
        var_update = data.std(0)
        # print('更新的均值为:{}\n标准差为:{}'.format(mean_update, var_update))

    def get_mask(self, mask_all, floor: int):
        segment = math.floor(floor / 3)
        color_range = (self.COLOR_MAP[segment] - 10, self.COLOR_MAP[segment] + 10)
        return ((color_range[0] <= mask_all) & (mask_all < color_range[1])).astype(np.uint8)

    def open_op(self, img_mask, kernel_size=3):
        return cv2.dilate(cv2.erode(img_mask, np.ones((kernel_size, kernel_size)), 3),
                          np.ones((kernel_size, kernel_size)), 3)

    def parse_image(self, img):
        """
        Return:
            {
                1: [outloop_pts1, ..., outloop_ptsN],
                ...
            }
        """
        # 楼栋面积阈值
        area_thr = (5 / 300 * img.shape[0]) ** 2  # 对应真实地块中的25㎡
        # 记录每种层高楼栋的轮廓列表
        floor_obj_map = {}

        for floor in self.floor_choice:
            floor_obj_map[floor] = []
            # 获取所有层高为floor的楼栋
            mask_floor = self.get_mask(img, floor)
            mask_floor = self.open_op(mask_floor)
            # seperate 每个楼栋
            components = cv2.connectedComponents(mask_floor)

            for label in range(1, components[0]):
                mask_build = (components[1] == label).astype(np.uint8)
                if mask_build.sum() > area_thr:
                    contour, _ = cv2.findContours(mask_build, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contour = np.array(contour).reshape(-1, 2).tolist()
                    if contour[0] != contour[-1]:
                        contour.append(contour[0])
                    floor_obj_map[floor].append(contour)
        return floor_obj_map

    def extract_outloop(self, img, type_flag='real'):
        if type_flag == 'real':
            img_binary = cv2.threshold(img, 250, 1, cv2.THRESH_BINARY_INV)[1]
        else:
            img_binary = cv2.adaptiveThreshold(img, 255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                               thresholdType=cv2.THRESH_BINARY_INV, blockSize=25, C=5)
        contour, hierachy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if hierachy.shape[1] > 1:  # 有多个轮廓
            contour = [c for c in contour if c.shape[0] > 2]  # 去除 点与线
            if len(contour) > 1:  # 只保留面积最大的
                contour = max(contour, key=lambda pts: cv2.contourArea(np.array(pts)))
        #     assert hierachy.shape[1] == 1, "找轮廓时可能由于外轮廓不连续，导致有多个"
        contour = np.array(contour).reshape(-1, 2).tolist()
        if contour[0] != contour[-1]:
            contour.append(contour[0])
        return contour

    def cal_condition(self, file, real_flag=True):
        """
        计算 [地块大小, 平均建筑层数, 地块密度, 建筑数量, 容积率]
        """
        img = cv2.imread(file)
        if len(img.shape) == 3:
            img = img[..., 0]
        build_info = self.parse_image(img)
        # 根据解析结果计算容积率
        field_area = cv2.contourArea(np.array(self.extract_outloop(img)).reshape(-1, 2)) * self.MAX_AREA / img.shape[0] / img.shape[1]
        volume_area = 0  # 计容面积
        cover_area = 0  # 占地面积
        num_builds = 0  # 建筑数量
        floor_list = []
        for floor, outloop_list in build_info.items():
            floor_list += [floor] * len(outloop_list)
            num_builds += len(outloop_list)
            for outloop in outloop_list:
                outloop_cover = cv2.contourArea(np.array(outloop))
                volume_area += outloop_cover * floor
                cover_area += outloop_cover

        volume_rate = volume_area / field_area
        density = cover_area / field_area
        floor_avg = float(np.mean(floor_list)) if floor_list else 0

        return [field_area, floor_avg, density, num_builds, volume_rate]

    def get_volume_rate(self, file):
        condition = self.get(file)
        vr = condition[-1]
        return (vr * self.condition_stdvar[-1]) + self.condition_mean[-1]

    def get(self, file):
        if not os.path.exists(file):
            return 0
        elif file in self.condition_dict:
            return self.condition_dict[file]
        else:
            # 生成并记录
            condition = self.cal_condition(file)
            # z-score
            condition = (np.array(condition) - self.condition_mean) / self.condition_stdvar
            self.condition_dict[file] = condition
            return condition

    def read_condition(self, input_list: list):
        return (np.array(input_list) * self.condition_stdvar + self.condition_mean).astype(np.float32)
