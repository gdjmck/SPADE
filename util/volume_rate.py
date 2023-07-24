import cv2
import math

import matplotlib.pyplot as plt
import numpy as np
import os
import json
from shapely.geometry import LineString

DEBUG = False


class Condition:
    def __init__(self, opt):
        self.condition_dict = {}
        self.condition_size = opt.condition_size
        self.condition_mask = self.get_condition_mask(self.condition_size)
        self.condition_name = np.array(['fieldSize', 'avgFloors', 'density', 'buildNum', 'volRat'])[self.condition_mask]
        try:
            with open(opt.condition_norm, 'r', encoding='utf-8') as f:
                condition_norm = json.load(f)
            self.condition_mean = np.array(condition_norm['mean'])
            self.condition_stdvar = np.array(condition_norm['stdvar'])
        except:
            self.condition_mean = np.array([0] * self.condition_size)
            self.condition_stdvar = np.array([1] * self.condition_size)
        # 过滤不需要的条件
        self.condition_mean = self.condition_mean[self.condition_mask]
        self.condition_stdvar = self.condition_stdvar[self.condition_mask]
        self.floor_choice = [1 + i * 3 for i in range(11)]  # 采样层数 [1, 4, 7, ..., 31]
        self.MAX_AREA = 90000
        self.COLOR_MAP = {i: 200 - i * 20 for i in range(11)}  # 层数与颜色的映射
        self.STANDARD_SIZE = 512

    def get_condition_mask(self, condition_size: int):
        """
        [地块大小, 平均建筑层数, 地块密度, 建筑数量, 容积率]
        :param condition_size:
        :return:
        """
        mask = [1] * 5
        if condition_size < 5:
            mask[0] = 0
        if condition_size < 4:
            mask[1] = 0
        if condition_size < 3:
            mask[2] = 0
        if condition_size < 2:
            mask[3] = 0
        return np.where(mask)

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

    def mask_dt_distance(self, mask, mask_dt):
        # 计算mask的点中距离最小值
        return mask_dt[mask].min()

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
        # 红线的距离变换矩阵，用于去除地块内部的颜色异常点
        field_outloop = self.extract_outloop(img)
        mask_field = np.ones_like(img, dtype=np.uint8)
        for pt_idx in range(len(field_outloop) - 1):
            pt_0, pt_1 = field_outloop[pt_idx], field_outloop[pt_idx + 1]
            cv2.line(mask_field, pt_0, pt_1, 0, lineType=cv2.LINE_AA)
            mask_field[pt_0[1], pt_0[0]] = 0
            mask_field[pt_1[1], pt_1[0]] = 0
        if DEBUG:
            plt.imsave('field.png', cv2.bitwise_not(mask_field))
        dtm = cv2.distanceTransform(mask_field * 255, cv2.DIST_L1, cv2.DIST_MASK_PRECISE)

        mask_realloc = np.zeros_like(img)
        floor_dict = {}
        floor_dt_dict = {}
        for floor in self.floor_choice:
            floor_obj_map[floor] = []
            # 获取所有层高为floor的楼栋
            mask_floor = self.get_mask(img, floor)
            mask_floor_clean = self.open_op(mask_floor)
            if DEBUG:
                plt.imsave('floor_{}.png'.format(floor), mask_floor_clean)
            floor_dict[floor] = mask_floor_clean
            floor_dt_dict[floor] = cv2.distanceTransform((1 - mask_floor_clean) * 255, cv2.DIST_L1,
                                                         cv2.DIST_MASK_PRECISE)
            mask_free = (mask_floor_clean == 0) & (mask_floor > 0)  # 被清掉的点
            # 判断是否为红线（距离红线小于3），不是红线则去除
            num_clear_comp, mask_clear_comp = cv2.connectedComponents(mask_free.astype(np.uint8), connectivity=4)
            for i in range(1, num_clear_comp):
                if self.mask_dt_distance(mask=mask_clear_comp == i, mask_dt=dtm) < 2:  # 到红线最短距离小于2，认为是红线的一部分
                    mask_free[mask_clear_comp == i] = 0
            mask_realloc |= mask_free

        # 对偏离的坐标重新赋值
        num_comp, label_mask_realloc = cv2.connectedComponents(mask_realloc.astype(np.uint8), connectivity=4)
        for realloc_id in range(1, num_comp):
            mask = label_mask_realloc == realloc_id
            # 找距离最近的楼栋进行分配
            target_floor_type, min_dist = None, np.inf
            for floor_type in floor_dt_dict.keys():
                d = self.mask_dt_distance(mask_dt=floor_dt_dict[floor_type], mask=mask)
                if d < min_dist:
                    min_dist = d
                    target_floor_type = floor_type

            if target_floor_type is not None:
                mask_realloc[mask] = 0
                floor_dict[target_floor_type][mask] = 1

        # 清空其余非前景区域
        mask_non_building = np.ones_like(img)
        # 去除红线区域
        mask_field_buffer = cv2.dilate(1-mask_field, kernel=np.ones((5, 5)))
        mask_non_building[mask_field_buffer > 0] = 0
        # 去除建筑
        for build_type_mask in floor_dict.values():
            mask_non_building[build_type_mask > 0] = 0
        # 清空图片的非前景区域
        img[mask_non_building > 0] = 255
        if DEBUG:
            plt.imsave('p1_mask.png', mask_non_building)
            plt.imsave('p1.png', img)

        for floor in floor_dict:
            # seperate 每个楼栋
            components = cv2.connectedComponents(floor_dict[floor], connectivity=4)

            for label in range(1, components[0]):
                mask_build = (components[1] == label).astype(np.uint8)
                if mask_build.sum() > area_thr:
                    contour, _ = cv2.findContours(mask_build, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contour = np.array(contour).reshape(-1, 2).tolist()
                    if contour[0] != contour[-1]:
                        contour.append(contour[0])
                    floor_obj_map[floor].append(contour)
                else:
                    # print('建筑面积太小{}被过滤'.format(mask_build.sum()))
                    mask_build[mask_field_buffer > 0] = 0
                    img[mask_build > 0] = 255

        return img, floor_obj_map

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
        if img.shape[0] != self.STANDARD_SIZE:
            fx = fy = self.STANDARD_SIZE / img.shape[0]
            img = cv2.resize(img, dsize=None, fx=fx, fy=fy)
        img, build_info = self.parse_image(img)
        # 根据解析结果计算容积率
        field_area = cv2.contourArea(np.array(self.extract_outloop(img)).reshape(-1, 2))
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

        return np.array([field_area, floor_avg, density, num_builds, volume_rate])[self.condition_mask].tolist()

    def get_volume_rate(self, file):
        condition = self.get(file)
        vr = condition[-1]
        return (vr * self.condition_stdvar[-1]) + self.condition_mean[-1]

    def get(self, file):
        """
        计算图片文件的容积率
        :param file:
        :return: 1-D numpy.array
        """
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

    def read_condition(self, input_list):
        return (np.array(input_list) * self.condition_stdvar + self.condition_mean).astype(np.float32)
