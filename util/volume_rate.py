import random
import traceback
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import json

DEBUG = False

def probe(val, stdvar, max_range: float=0.1):
    seed = random.randint(1, 10) / 10
    sign = random.randint(0, 1) == 1
    return val + sign * seed * max_range / stdvar

class Condition:
    condition_dict = {'s': 'fieldSize', 'v': 'volRat', 'n': 'buildNum', 'f': 'avgFloors', 'd': 'density'}

    def __init__(self, opt):
        self.opt = opt
        self.condition_dict = {}
        self.condition_size = opt.condition_size
        # 所有条件label必须在condition_dict中
        assert all([c in Condition.condition_dict for c in list(opt.condition_order)])
        self.condition_name = np.array([Condition.condition_dict[c] for c in list(opt.condition_order)])
        self.condition_mask = np.ones(len(Condition.condition_dict)).astype(bool)
        self.condition_mask[self.condition_size:] = False
        try:
            with open(opt.condition_norm, 'r', encoding='utf-8') as f:
                condition_norm = json.load(f)
            self.condition_mean = np.array(condition_norm['mean'])
            self.condition_stdvar = np.array(condition_norm['stdvar'])
        except:
            self.condition_mean = np.array([0] * self.condition_size)
            self.condition_stdvar = np.array([1] * self.condition_size)
        # 条件的均值标准差按condition_order进行重新排序
        self.reorder_index_list = self.condition_order_mask()
        self.condition_mean = self.reorder(self.condition_mean)
        self.condition_stdvar = self.reorder(self.condition_stdvar)

        # 原始json数据计算条件
        self.condition_json = None
        try:
            if os.path.exists(opt.condition_json):
                with open(opt.condition_json, 'r') as f:
                    condition_json = json.load(f)
                # 将列表转换为_id为key的字典
                self.condition_json = {}  # id: [boundary, buildings]
                for item in condition_json:
                    self.condition_json[item['_id']] = {key: item[key] for key in ['boundary', 'buildings']}
        except:
            print(traceback.format_exc())

        # 过滤不需要的条件
        self.condition_mean = self.condition_mean[self.condition_mask]
        self.condition_stdvar = self.condition_stdvar[self.condition_mask]
        self.floor_choice = [1 + i * 3 for i in range(11)]  # 采样层数 [1, 4, 7, ..., 31]
        self.MAX_AREA = 90000
        self.COLOR_MAP = {i: 200 - i * 20 for i in range(11)}  # 层数与颜色的映射
        self.STANDARD_SIZE = 512

    def reorder(self, condition: np.array):
        return condition[self.reorder_index_list]

    def condition_order_mask(self):
        """
        [地块大小, 平均建筑层数, 地块密度, 建筑数量, 容积率]
        :param condition_size:
        :return: opt.condition_order对默认条件顺序的索引
        """
        default_order = list('sfdnv')
        return [default_order.index(c) for c in list(self.opt.condition_order)]

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
        """
        按层数获取相应的掩码
        :param mask_all:
        :param floor:
        :return:
        """
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

    def read_condition_from_json(self, file):
        split_char = '/' if '/' in file else '\\'
        fid = file.rsplit(split_char, 1)[-1].rsplit('.', 1)[0]
        if self.condition_json and fid in self.condition_json:
            raw_data = self.condition_json[fid]
            field_size = cv2.contourArea(np.array(raw_data['boundary'], dtype=np.float32))
            condition_values = []
            for c in self.opt.condition_order[:self.opt.condition_size]:
                if c == 's':  # 地块大小
                    condition_values.append(field_size)
                elif c == 'f':  # 平均层数
                    avg_floor = np.array([build['floor'] for build in raw_data['buildings']]).mean()
                    condition_values.append(avg_floor)
                elif c == 'd':  # 密度
                    cover_area = sum([cv2.contourArea(np.array(build['coords'], dtype=np.float32)) for build in raw_data['buildings']]) \
                                 / field_size
                    condition_values.append(cover_area)
                elif c == 'n':  # 建筑数量
                    num_builds = len(raw_data['buildings'])
                    condition_values.append(num_builds)
                elif c == 'v':  # 容积率
                    volume_ratio = sum([cv2.contourArea(np.array(build['coords'], dtype=np.float32)) * build['floor']
                                        for build in raw_data['buildings']]) / field_size
                    condition_values.append(volume_ratio)
            return condition_values
        else:
            return False


    def cal_condition(self, file, real_flag=True):
        """
        计算原始条件 [地块大小, 平均建筑层数, 地块密度, 建筑数量, 容积率]
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

        condition_array = np.array([field_area, floor_avg, density, num_builds, volume_rate])
        condition_array = self.reorder(condition_array)

        return condition_array[self.condition_mask].tolist()


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
            condition = None
            if self.condition_json:  # 从json中读取
                condition = self.read_condition_from_json(file)
            if not condition:  # 从图片中读取
                condition = self.cal_condition(file)
            # z-score
            condition = (np.array(condition) - self.condition_mean) / self.condition_stdvar
            self.condition_dict[file] = condition
            return condition

    def read_condition(self, input_list):
        return (np.array(input_list) * self.condition_stdvar + self.condition_mean).astype(np.float32)

if __name__ == '__main__':
    # 统计数据集的条件均值与方差
    from options.train_options import TrainOptions
    # parse options
    opt = TrainOptions().parse()
    condition_logger = Condition(opt)
    condition_logger.condition_json