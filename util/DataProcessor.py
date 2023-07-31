import pandas as pd
import geopandas as gpd
import json
from shapely.geometry import Point, Polygon, LineString
from shapely.affinity import translate
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import copy
import cv2

MAX_AREA = 90000
MAX_SIZE = 300
MIN_AREA = 1500
CONTAIN_LEAST_BUILD = 4

COLOR_MAP = {i: 200 - i * 20 for i in range(11)}


def get_build_color(floor: int, max_floor=32):
    """
        楼栋层数 -> 颜色编号(matplotlib)
    """
    floor = max(0, min(max_floor, floor))
    color_seg = math.floor(floor / 3)
    color_val = COLOR_MAP[color_seg]
    color_code = hex(color_val).upper()[2:]
    if len(color_code) == 1:
        color_code = '0' + color_code

    return "#" + color_code * 3


def intersection(build_poly, field_poly):
    """
    计算楼栋数据build的Polygon在地块内的部分，当少于50%面积在地块内时舍弃
    Args:
        build_poly: Polygon
        field_poly: Polygon
    """
    build_its_poly = build_poly.intersection(field_poly)
    if build_its_poly.area / build_poly.area < 0.5:
        return None

    if isinstance(build_its_poly, Polygon):
        return build_its_poly
    else:
        return build_poly


def is_valid_field(field_item):
    """
    地块面积小于MAX_AREA ㎡，宽高均小于MAX_SIZE米
    """
    poly = field_item.geometry
    area = float(poly.area)
    if area > MAX_AREA or area < MIN_AREA:
        return False
    bounds = poly.bounds
    width = float(bounds['maxx']) - float(bounds['minx'])
    height = float(bounds['maxy']) - float(bounds['miny'])
    if width > MAX_SIZE or height > MAX_SIZE:
        return False
    return True


def get_floor(build_item):
    return int(build_item.get('FLOOR',
                                build_item.get('floor',
                                               build_item.get('height',
                                                              build_item.get('Elevation',
                                                                             pd.Series(np.array([0.])))) / 3.0)).values)


def get_bias(field_item):
    """
    地块中心到目标中心（MAX_SIZE//2， MAX_SIZE//2）的偏移量
    """
    field_poly = field_item.geometry.values[0]
    field_centroid = field_poly.centroid
    minx, miny, maxx, maxy = field_poly.bounds
    bias_x, bias_y = MAX_SIZE / 2 - field_centroid.x, MAX_SIZE / 2 - field_centroid.y
    # 保证地块轮廓不超出显示区域
    bias_x = min(max(-minx, bias_x), MAX_SIZE - maxx)
    bias_y = min(max(-miny, bias_y), MAX_SIZE - maxy)

    return {'xoff': bias_x,
            'yoff': bias_y}


class ImageSaver:
    def __init__(self, dir_field, dir_arch):
        # 地块图片文件夹
        self.dir_field = dir_field
        if not os.path.exists(dir_field):
            os.makedirs(dir_field)
        # 地块建筑文件夹
        self.dir_arch = dir_arch
        if not os.path.exists(dir_arch):
            os.makedirs(dir_arch)

    def plot_field_reference(self, ax):
        gpd.GeoSeries(LineString([[0, 0], [0, MAX_SIZE], [MAX_SIZE, MAX_SIZE], [MAX_SIZE, 0], [0, 0]])).plot(ax=ax,
                                                                                                             facecolor='none',
                                                                                                             edgecolor='none')

    def plot_build(self, obj, color, ax):
        obj.plot(ax=ax, facecolor=color, edgecolor='none', aspect='equal')

    def is_field_exists(self, field_id):
        return os.path.exists(os.path.join(self.dir_field, '{}.jpg'.format(field_id)))

    def save(self, field_id, data_field, data_build=None):
        """
        Args:
            data_build: list, elements of shape: (floor: int, poly: GeoSeries)
            data_field: GeoSeries
        """

        fig = plt.figure(figsize=(10.24, 10.24), dpi=100)  # 1024*1024
        ax = fig.add_subplot(1, 1, 1)
        plt.axis([0, MAX_SIZE, 0, MAX_SIZE])
        #         plt.axis('auto')
        self.plot_field_reference(ax)
        data_field.plot(ax=ax, facecolor='none', edgecolor='black')

        if data_build:
            # 根据楼栋层数画轮廓
            for build_item in data_build:
                floor, poly_gs = build_item
                self.plot_build(poly_gs, get_build_color(floor), ax)

        ax.set_axis_off()
        plt.savefig(os.path.join(self.dir_arch if data_build else self.dir_field, "{}.png".format(field_id)), dpi=100,
                    pad_inches=0)
        plt.close()


####################### 数据集划分 ##################
# 楼层离散化，为了减少颜色的种类
floor_choice = [1 + i * 3 for i in range(11)]


def open_op(img_mask):
    return cv2.dilate(cv2.erode(img_mask, (3, 3), 3), (3, 3), 3)


def volume_rate(build_info, img_raw):
    # 根据解析结果计算容积率
    field_area = Polygon(np.array(extract_outloop(img_raw)).reshape(-1, 2)).area
    volume_area = 0
    for floor, outloop_list in build_info.items():
        for outloop in outloop_list:
            volume_area += Polygon(outloop).area * floor
    return volume_area / field_area


def parse_image(img):
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

    for floor in floor_choice:
        floor_obj_map[floor] = []
        # 获取所有层高为floor的楼栋
        mask_floor = get_mask(img, floor)
        mask_floor = open_op(mask_floor)
        # seperate 每个楼栋
        components = cv2.connectedComponents(mask_floor)

        for label in range(1, components[0]):
            mask_build = (components[1] == label).astype(np.uint8)
            if mask_build.sum() > area_thr:
                _, contour, _ = cv2.findContours(mask_build, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour = np.array(contour).reshape(-1, 2).tolist()
                if contour[0] != contour[-1]:
                    contour.append(contour[0])
                floor_obj_map[floor].append(contour)
    return floor_obj_map


def plot_parsing(result):
    outloops, color = [], []
    for floor, loop_list in result.items():
        for loop in loop_list:
            outloops.append(LineString(loop))
            color.append([floor * 50] * 3)
    ax = plt.gca()
    ax.invert_yaxis()
    GeoSeries(outloops).plot(ax=ax)


def extract_outloop(img):
    img_contour, contour, _ = cv2.findContours(cv2.threshold(img, 200, 1, cv2.THRESH_BINARY_INV)[1],
                                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = np.array(contour).reshape(-1, 2).tolist()
    return contour


def get_area(outloop, field_pix_area: int):
    outloop_area = Polygon(np.array(outloop).reshape(-1, 2)).area
    return outloop_area / field_pix_area * MAX_AREA


def get_mask(mask_all, floor: int):
    segment = math.floor(floor / 3)
    color_range = (COLOR_MAP[segment] - 10, COLOR_MAP[segment] + 10)
    return ((color_range[0] <= mask_all) & (mask_all < color_range[1])).astype(np.uint8)
