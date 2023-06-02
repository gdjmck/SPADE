import os
import cv2
import math
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString
from geopandas import GeoSeries
from PIL import Image


def extract_outloop(img):
    contour, hierachy = cv2.findContours(cv2.threshold(img, 250, 1, cv2.THRESH_BINARY_INV)[1],
                                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if hierachy.shape[1] > 1:  # 有多个轮廓
        contour = [c for c in contour if c.shape[0] > 2]  # 去除 点与线
        if len(contour) > 1:  # 只保留面积最大的
            contour = max(contour, key=lambda pts: Polygon(pts).area)
    #     assert hierachy.shape[1] == 1, "找轮廓时可能由于外轮廓不连续，导致有多个"
    contour = np.array(contour).reshape(-1, 2).tolist()
    if contour[0] != contour[-1]:
        contour.append(contour[0])
    return contour

def open_op(img_mask):
    return cv2.dilate(cv2.erode(img_mask, np.ones((3, 3)), 3), np.ones((3, 3)), 3)

class PostProcess:
    def __init__(self, tolerance=1.0):
        self.tolerance = tolerance
        self.field_loop = None
        self.img = None
        self.mask = None
        self.num_comp = -1
        self.bg_id = -1
        self.building_list = []

    def clear(self):
        self.field_loop = None
        self.img = None
        self.mask = None
        self.num_comp = -1
        self.bg_id = -1
        self.building_list = []

    def process(self, img):
        self.clear()
        self.img = img

        self.field_loop = LineString(extract_outloop(img))
        # img_bin = cv2.adaptiveThreshold(img, 255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        #                                 thresholdType=cv2.THRESH_BINARY_INV, blockSize=25, C=5)
        img_bin = cv2.threshold(img, 250, 1, cv2.THRESH_BINARY_INV)[1]
        flat_area = open_op(img_bin) & img_bin
        self.num_comp, self.mask = cv2.connectedComponents(flat_area, connectivity=4)
        self.bg_id = self.mask[0, 0]

        # 提取每个楼栋的轮廓
        for i in range(self.num_comp):
            if i == self.bg_id:
                continue
            self.extract_build(self.mask == i)

    def extract_build(self, mask):
        # val = np.median(self.img[mask])
        loop, contours_info = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        loop = self.simplify(loop[0])
        # 楼栋轮廓不能与红线相交
        loop = self.seperate(loop)

        self.building_list.append(loop)

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