import cv2
import math
import numpy as np
import os



class VolumeRate:
    def __init__(self):
        self.volume_dict = {}
        self.floor_choice = [1+i*3 for i in range(11)]  # 采样层数 [1, 4, 7, ..., 31]
        self.MAX_AREA = 90000
        self.COLOR_MAP = {i: 200 - i * 20 for i in range(11)}  # 层数与颜色的映射

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

    def extract_outloop(self, img):
        contour, hierachy = cv2.findContours(cv2.threshold(img, 250, 1, cv2.THRESH_BINARY_INV)[1],
                                                          cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if hierachy.shape[1] > 1:  # 有多个轮廓
            contour = [c for c in contour if c.shape[0] > 2]  # 去除 点与线
            if len(contour) > 1:  # 只保留面积最大的
                contour = max(contour, key=lambda pts: cv2.contourArea(np.array(pts)))
        #     assert hierachy.shape[1] == 1, "找轮廓时可能由于外轮廓不连续，导致有多个"
        contour = np.array(contour).reshape(-1, 2).tolist()
        if contour[0] != contour[-1]:
            contour.append(contour[0])
        return contour

    def volume_ratio(self, file):
        img = cv2.imread(file)
        if len(img.shape) == 3:
            img = img[..., 0]
        build_info = self.parse_image(img)
        # 根据解析结果计算容积率
        field_area = cv2.contourArea(np.array(self.extract_outloop(img)).reshape(-1, 2))
        volume_area = 0
        for floor, outloop_list in build_info.items():
            for outloop in outloop_list:
                volume_area += cv2.contourArea(np.array(outloop)) * floor
        return volume_area / field_area

    def get(self, file):
        if not os.path.exists(file):
            return 0
        elif file in self.volume_dict:
            return self.volume_dict[file]
        else:
            # 生成并记录
            volume_ratio = self.volume_ratio(file)
            self.volume_dict[file] = volume_ratio
            return volume_ratio