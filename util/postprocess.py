import os
import cv2
import math
import glob
import shutil
import json
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from pyproj import CRS
from shapely.geometry import Polygon, LineString
from shapely.affinity import scale, translate
from geopandas import GeoSeries
from PIL import Image
from util.geo_util import coord_to_longtitude


def extract_outloop(img):
    def get_area(pts):
        try:
            return Polygon(pts).area
        except ValueError:
            return 0

    contour, hierachy = cv2.findContours(cv2.threshold(img, 250, 1, cv2.THRESH_BINARY_INV)[1],
                                         cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if hierachy.shape[1] > 1:  # 有多个轮廓
        contour = [c for c in contour if c.shape[0] > 2]  # 去除 点与线
        if len(contour) > 1:  # 只保留面积最大的
            contour = max(contour, key=lambda pts: get_area(pts))
    #     assert hierachy.shape[1] == 1, "找轮廓时可能由于外轮廓不连续，导致有多个"
    contour = np.array(contour).reshape(-1, 2).tolist()
    if contour[0] != contour[-1]:
        contour.append(contour[0])
    return contour


def open_op(img_mask):
    return cv2.dilate(cv2.erode(img_mask, np.ones((3, 3)), 3), np.ones((3, 3)), 3)


class PostProcess:
    def __init__(self, max_size, tolerance=1.0, field_size=300):
        """

        :param max_size:
        :param tolerance:
        :param field_size: 图片对应真实地块的尺寸（米）
        """
        self.tolerance = tolerance
        self.field_size = field_size
        self.max_size = max_size  # 用于y轴反转
        self.minBuildArea = 25
        self.segment_to_color = {0: 200,
                                 1: 180,
                                 2: 160,
                                 3: 140,
                                 4: 120,
                                 5: 100,
                                 6: 80,
                                 7: 60,
                                 8: 40,
                                 9: 20,
                                 10: 0}
        self.init()

    def init(self):
        self.clear()

    def clear(self):
        self.field_loop = None
        self.img = None
        self.mask = None
        self.num_comp = -1
        self.bg_id = -1
        self.building_list = []
        self.floor_list = []
        self.build2maskId = {}
        self.scale = 0  # 一个像素代表的面积

    def color2floor(self, color_val: int):
        # 高度映射颜色 color_val = 220 - 2 * floor
        # floor height = 3.0
        return max(1, math.ceil((220 - color_val) / 6))

    def set_scale(self):
        h, w = self.img.shape[:2]
        assert h == w
        self.scale = (self.field_size / h) ** 2

    def process(self, img, type='real'):
        """
        从图片解析建筑，得到建筑轮廓与层数（color2floor）
        结果:
            轮廓list：self.building_list
            层数list: self.floor_list
        """
        self.clear()
        self.img = img
        self.set_scale()

        self.field_loop = LineString(extract_outloop(img))
        if type == 'real':
            img_bin = cv2.threshold(img, 250, 1, cv2.THRESH_BINARY_INV)[1]
        else:
            img_bin = cv2.adaptiveThreshold(img, 255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                            thresholdType=cv2.THRESH_BINARY_INV, blockSize=25, C=5)

        flat_area = open_op(img_bin) & img_bin
        self.num_comp, self.mask = cv2.connectedComponents(flat_area, connectivity=4)
        self.bg_id = self.mask[0, 0]

        # 提取每个楼栋的轮廓
        for i in range(self.num_comp):
            if i == self.bg_id:
                continue
            mask = self.mask == i
            cover_area = mask.sum()
            # 过滤面积小于25㎡
            if cover_area * self.scale < self.minBuildArea:
                continue
            if self.contain_multiple_object(mask):
                pass
                # print('轮廓索引{}包含多个建筑'.format(len(self.building_list)))

            self.extract_build(self.mask == i)

    def contain_multiple_object(self, mask):
        color_vals = self.img[mask]
        color_stats = Image.fromarray(color_vals).getcolors(color_vals.shape[0])
        if len(color_stats) < 2:
            return False
        # 按频数高到低排序
        color_stats = sorted(color_stats, key=lambda item: item[0], reverse=True)
        # 颜色数量比例不超过5: 1 && 颜色差值大于20
        color_most, color_second_most = color_stats[:2]
        if color_most[0] / color_second_most[0] <= 5 and abs(color_most[1] - color_second_most[1]) > 20:
            # 暂且认为只会有
            return True
        else:
            return False, None

    def add_build(self, build_loop: list, mask):
        mask_val = np.median(self.mask[mask])
        self.build2maskId[len(self.building_list)] = mask_val  # 用于调试时查看楼栋对应的mask，self.mask == build.index
        self.building_list.append(build_loop)
        self.floor_list.append(self.color2floor(np.median(self.img[mask])))

    def extract_build(self, mask):
        # val = np.median(self.img[mask])
        loop, contours_info = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        loop = self.simplify(loop[0])
        # 楼栋轮廓不能与红线相交
        loop = self.seperate(loop)
        """
        # 判断楼栋轮廓内是否只有一种楼
        multi_build, colors = self.is_multipart(mask)
        multi_build = False  # 先屏蔽裙楼
        if multi_build:
            # 区分不同楼栋
            # 假设存在不同楼栋的情况只有内部包含另外一种楼
            mask1 = mask & (self.img == colors[0][1])
            mask2 = mask & (self.img == colors[1][1])
            hole1 = PostProcess.find_hole(mask1)
            hole2 = PostProcess.find_hole(mask2)
            assert any([len(hole1), len(hole2)])

            mask_w_hole = mask1
            mask_wo_hole = mask2
        else:
            self.add_build(loop, mask=mask)
        """
        self.add_build(loop, mask=mask)

        return loop

    def to_output(self, real_center, standard_size, coord2longlat=False):
        """
        构造返回数据
        """
        outloop = self.building_list
        field_size = (self.field_loop.bounds[2]-self.field_loop.bounds[0],
                      self.field_loop.bounds[3]-self.field_loop.bounds[1])
        scaler = min([s_real / s_gen for s_gen, s_real in zip(field_size, standard_size)], key=lambda v: abs(v-1))
        field_poly = Polygon(self.field_loop)
        bias_dict = {'xoff': real_center[0]-field_poly.centroid.x,
                     'yoff': real_center[1]-field_poly.centroid.y}
        # 恢复偏移
        for i in range(len(outloop)):
            outloop_ls = LineString(np.array(outloop[i]))
            outloop_ls = scale(translate(outloop_ls, **bias_dict),
                               scaler, scaler, origin=list(real_center)+[0.0])
            outloop[i] = np.array(outloop_ls.coords).tolist()
            if coord2longlat:
                # 坐标转回经纬度
                outloop[i] = [coord_to_longtitude(*coord) for coord in outloop[i]]
        # # 把生成的地块轮廓也加上
        # field_outloop = scale(translate(self.field_loop, **bias_dict), scaler, scaler, origin=list(real_center)+[0.0])
        # print('生成的地块轮廓重心:{}'.format(Polygon(field_outloop).centroid))
        # outloop.append(np.array(field_outloop.coords).tolist())
        return {'outloop': outloop, 'floor': self.floor_list}

    ################# 工具函数 #############################

    def is_multipart(self, mask):
        """
        判断图像img的掩码mask中是否有多种颜色（多种建筑）
        """
        color_list = self.img[mask]
        colors = Image.fromarray(color_list).getcolors(color_list.shape[0])
        # 取数量最多的2个颜色
        colors = sorted(colors, key=lambda item: item[0], reverse=True)
        color_most, color_2nd_most = colors[:2]
        # 最多颜色数量 / 第二多颜色数量 < 5 && abs(最多颜色 - 第二多颜色) > 30 (跨度超过1个色域)
        if color_most[0] / color_2nd_most[0] < 5 and abs(color_most[1] - color_2nd_most[1]) > 30:
            return True, colors
        else:
            return False, colors

    @classmethod
    def find_hole(cls, mask):
        hole_list = []
        try:
            pts, hierachy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        except TypeError:
            mask = mask.astype(np.uint8)
            pts, hierachy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        hierachy = hierachy.reshape(-1, 4)
        children_index = np.where(hierachy[:, -1] != -1)[0]
        for child_index in children_index:
            child = pts[child_index].reshape(-1, 2)
            if child.shape[0] < 3:
                continue
            child_area = Polygon(child).area

            parent_index = hierachy[child_index][3]
            if hierachy[parent_index][-1] != -1:  # 只取最外层的子轮廓
                print('非外层的子轮廓')
                continue
            parent = pts[parent_index].reshape(-1, 2)
            parent_area = Polygon(parent).area
            child_area_over_parent = child_area / parent_area
            print('child面积占比{:.2f}'.format(child_area_over_parent))
            if child_area_over_parent > 0.2:
                child = child.tolist()
                if child[0] != child[-1]:
                    child.append(child[0])
                hole_list.append(child)
        print('内部空洞个数:{}'.format(len(hole_list)))
        return hole_list

    def seperate(self, loop: list):
        """
        使楼栋轮廓与地块红线分离开
        """
        loop_poly = Polygon(loop).buffer(-0.01)
        if not loop_poly.intersects(self.field_loop):
            return loop
        while loop_poly.distance(self.field_loop) <= 0.01:
            loop_poly = loop_poly.buffer(-0.01)
        return list(loop_poly.exterior.simplify(2 * self.tolerance).coords)

    def simplify(self, contour):
        contour = contour.reshape(-1, 2).tolist()
        if contour[0] != contour[-1]:
            contour.append(contour[0])
        # 最小距离为toerlance
        contour = LineString(contour).simplify(self.tolerance)
        contour = list(contour.coords)

        return contour

    def plot(self):
        ax = plt.gca()
        ax.invert_yaxis()
        GeoSeries([self.field_loop] + [LineString(build) for build in self.building_list]).plot(ax=ax)


def to_geojson(data: dict):
    """
    排楼结果转成shp文件
    :param layout_result: json 路径
    :param shpfiles_dir: 存储shp文件位置
    :param shpfiles_name:shp名字
    :return:
    """
    geo = {
        'geometry': [Polygon(outloop) for outloop in data['outloop']],
        'floor': data['floor']
    }

    s = gpd.GeoDataFrame(geo, crs="EPSG:3857")
    # 定义目标坐标系
    target_crs = CRS.from_epsg(4326)
    # 将原始数据集的坐标系转换为目标坐标系
    s = s.to_crs(target_crs)
    return json.loads(s.to_json())


if __name__ == '__main__':
    fig = plt.figure(figsize=(10.24, 10.24), dpi=100)
    processor = PostProcess()
    img = cv2.imread(r'd:\Documents\aisr\GeosRelate\dataset_style3_slim\arch_FS\1903.jpg')
    img_gen = np.mean(img, -1).astype(np.uint8)
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(img)
    # 后处理
    processor.process(img_gen)
    ax = fig.add_subplot(1, 2, 2)
    processor.plot()
