import glob
import json
import matplotlib
matplotlib.use('Agg')

import torch
import numpy as np
import scipy
from options.base_options import BaseOptions
from models.pix2pix_model import Pix2PixModel
from models.unet_model import UNetModel
from data.base_dataset import BaseDataset, get_params, get_transform, convert_label_image
from util.util import tensor2im, save_image
from util.DataProcessor import ImageSaver, MAX_SIZE
from util.postprocess import PostProcess, to_geojson
from util.geo_util import longtitude_to_coord, coord_to_longtitude
from shapely.geometry import Polygon
from shapely.affinity import translate
import geopandas as gpd
import torchvision.transforms as transforms
from PIL import Image
import argparse
import pickle
import random
import web
import os
import requests

class EvalOptions(BaseOptions):
    def __init__(self, opt_file):
        super(EvalOptions, self).__init__()
        self.opt_file = opt_file
        self.isTrain = False

    def load_options(self, opt):
        return pickle.load(open(self.opt_file, 'rb'))

    def gather_options(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)
        opt, _ = parser.parse_known_args()
        parser = self.update_options_from_file(parser, opt)
        opt = parser.parse_args()
        self.parser = parser
        return opt


class DataPrep:
    def __init__(self, vae_ref_file=None):
        self.vae_ref_file = vae_ref_file
        self.image_size = 256
        self.label_transform = transforms.Compose([transforms.Resize(self.image_size, interpolation=Image.NEAREST),
                                                   transforms.ToTensor()])
        self.image_transform = transforms.Compose([transforms.Resize(self.image_size, interpolation=Image.BICUBIC),
                                                   transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5),
                                                                                               (0.5, 0.5, 0.5))])
        if self.vae_ref_file:
            vae_ref = Image.open(self.vae_ref_file).convert('RGB')
            vae_ref = self.image_transform(vae_ref)
            if vae_ref.size(0) > 1:
                vae_ref = vae_ref[:1]
            self.vae_ref = vae_ref
        else:
            self.vae_ref = None

    def preprocess_image(self, field_image_file):
        label = Image.open(field_image_file).convert('L')
        label = Image.fromarray(convert_label_image(np.array(label)))
        label_tensor = self.label_transform(label)

        vr = 1.5 + random.random()  # 容积率随机，范围在[1.5, 2.5]
        vr = torch.tensor(vr)

        return {'label': label_tensor, 'vr': vr, 'image': self.vae_ref}



urls = (
    '/generateLayout', 'GenerateLayout',
)

class WebApp(web.application):
    def run(self, port=8080, *middleware):
        func = self.wsgifunc(*middleware)
        return web.httpserver.runsimple(func, ('0.0.0.0', port))

app = WebApp(urls, globals())

class GenerateLayout:
    def __init__(self):
        self.temp_dir = './db/'
        self.image_saver = ImageSaver(self.temp_dir, self.temp_dir)
        self.postprocessor = PostProcess(256)
        self.scale = MAX_SIZE / 256  # 256是模型是输出尺寸
        self.batchSize = 4
        self.interface_longlat2coords = 'http://10.1.15.61:8080/util/longlat2coords'
        self.interface_coords2longlat = 'http://10.1.15.61:8080/util/coords2longlat'

    def get_bias(self, field_poly):
        field_centroid = field_poly.centroid
        minx, miny, maxx, maxy = field_poly.bounds
        bias_x, bias_y = MAX_SIZE / 2 - field_centroid.x, MAX_SIZE / 2 - field_centroid.y
        # 保证地块轮廓不超出显示区域
        bias_x = min(max(-minx, bias_x), MAX_SIZE - maxx)
        bias_y = min(max(-miny, bias_y), MAX_SIZE - maxy)
        bias = {'xoff': bias_x,
                'yoff': bias_y}
        return bias

    def get_file(self, name):
        return os.path.join(self.temp_dir, '{}.png'.format(name))

    def longlat2coords(self, data):
        response = requests.post(self.interface_longlat2coords, {'data': json.dumps(data)})
        data = json.loads(response.content)
        return data

    def coords2longlat(self, data):
        response = requests.post(self.interface_coords2longlat, {'data': json.dumps(data)})
        data = json.loads(response.content)
        return data

    def preprocess_list(self, data):
        field_data = {}
        for i, field_outloop in enumerate(data):
            field_outloop = np.array(field_outloop).reshape(-1, 2).tolist()
            # 转坐标
            coords = [longtitude_to_coord(*pt) for pt in field_outloop]
            field_data[i] = coords
        return field_data

    def preprocess(self, data):
        """
        经纬度转投影坐标
        Return:
            field_data: dict {field_id: [[x, y]]}
        """
        field_data = {}
        for k, field in enumerate(data['features']):
            try:
                field_name = field['properties']['DKBM']
                geo_pts = field['geometry']['coordinates']
            except:
                field_name = str(k)
                geo_pts = [field]
            for i, field_outloop_list in enumerate(geo_pts):
                for j, field_outloop in enumerate(field_outloop_list):
                    field_outloop = np.array(field_outloop).reshape(-1, 2).tolist()
                    # 转坐标
                    coords = [longtitude_to_coord(*pt) for pt in field_outloop]
                    if i == 0 and j == 0:
                        field_data[field_name] = coords
                    else:
                        field_data[field_name + '_{}_{}'.format(i, j)] = coords
        return field_data

    def request_from_url(self, url):
        return json.loads(bytes.decode(requests.get(url, timeout=5).content))


    def POST(self):
        web.header('Access-Control-Allow-Origin', '*')
        try:
            data = web.input()['redLine']
            data = json.loads(data)
        except KeyError:
            return {'code': 400, 'msg': '数据格式有误'}
        if len(data) == 0:
            return []
        origin_size = len(data)
        data = self.preprocess_list(data)

        # data = web.input()['url']
        # data = self.request_from_url(data)
        # if len(data['features']) > 5:
        #     data['features'] = data['features'][:5]
        # data = self.preprocess(data)

        # if len(data) > 5:
        #     data_new = {}
        #     for i, key in enumerate(data):
        #         if i >= 5:
        #             break
        #         data_new[key] = data[key]
        #     data = data_new

        # # 判断是否经纬度坐标
        # needs_conversion = False
        # sample = np.array(data[0]).reshape(-1, 2)
        # if Polygon(sample).area < 10:
        #     needs_conversion = True
        # if needs_conversion:
        #     size_list = []
        #     data_concat = []
        #     for item in data:
        #         item = np.array(item).reshape(-1, 2)
        #         size_list.append(item.shape[0])
        #         data_concat.extend(item.tolist())
        #     data_concat = self.longlat2coords(data_concat)
        #     # 数据重构
        #     for i in range(len(data)):
        #         data[i] = data_concat[:size_list[i]]
        #         data_concat = data_concat[size_list[i]:]


        # 生成地块图片
        input_dict_list = []
        real_centers = []
        field_size = []
        field_key = []
        fail_field_dict = {}
        for field_name in data:
            field_poly = Polygon(np.array(data[field_name]).reshape(-1, 2))
            bias = self.get_bias(field_poly)
            real_centers.append([field_poly.centroid.x, field_poly.centroid.y])
            field_size.append((field_poly.bounds[2]-field_poly.bounds[0],
                               field_poly.bounds[3]-field_poly.bounds[1]))
            print('入参图形的重心:{}'.format(real_centers[-1]))
            field_poly = translate(field_poly, **bias)
            self.image_saver.save(field_name, gpd.GeoSeries(field_poly))
            try:
                input_dict_list.append(processor.preprocess_image(self.get_file(field_name)))
            except:
                # 舍弃当前数据
                field_size.pop()
                real_centers.pop()
                # 记录无法处理的数据
                fail_field_dict[field_name] = "无法处理地块"
                continue

            field_key.append(field_name)
        num_fields = len(field_key)

        # 构造batch data
        input_semantics_list, reference_image_list, vr_list = [], [], []
        for input_dict in input_dict_list:
            label = input_dict['label'].long()
            _, h, w = label.size()
            input_semantics = torch.FloatTensor(2, h, w).zero_()
            input_semantics = input_semantics.scatter_(0, label, 1.0)
            input_semantics_list.append(input_semantics.unsqueeze(0))
            vr_list.append(input_dict['vr'].unsqueeze(0))
            reference_image_list.append(input_dict['image'].unsqueeze(0))
        if len(input_semantics_list) == 0:
            return []
        # 数据拼接
        input_semantics_tensor = torch.cat(input_semantics_list, 0).cuda()
        reference_image_tensor = torch.cat(reference_image_list, 0).cuda()
        vr_tensor = torch.cat(vr_list, 0).cuda()

        # forward
        with torch.no_grad():
            b, c, h, w = input_semantics_tensor.size()
            gen_image, _ = model.generate_fake(input_semantics_tensor, reference_image_tensor, volume_ratio=vr_tensor)
            for i in range(num_fields):
                save_image(tensor2im(gen_image[i]), './gen_{}.png'.format(field_key[i]))

        result = []
        # 后处理
        for i in range(num_fields):
            img = tensor2im(gen_image[i])
            img = img[::-1, ...]  # inverse y-axis
            try:
                self.postprocessor.process(img[..., 0], 'fake')
            except:
                print('{}处理失败'.format(field_key[i]))
                result.append('后处理失败，请重新生成')
                continue
            # 构造返回数据
            output = self.postprocessor.to_output(real_centers[i], field_size[i], coord2longlat=False)
            if output is None:
                result.append('后处理失败，请重新生成')
                continue
            # 转换为geojson
            output = to_geojson(output)
            # 恢复偏移
            result.append(output)
        # if needs_conversion:
        #     size_list = []
        #     data_concat = []
        #     for field in result:
        #         field_list = []
        #         for build in field['outloop']:
        #             outloop = np.array(build).reshape(-1, 2)
        #             field_list.append(outloop.shape[0])
        #             data_concat.extend(outloop.tolist())
        #         size_list.append(field_list)
        #     data_concat = self.coords2longlat(data_concat)
        #     # 数据重构
        #     for i in range(len(result)):
        #         for j in range(len(size_list[i])):
        #             batch_size = size_list[i][j]
        #             result[i]['outloop'][j] = data_concat[:batch_size]
        #             data_concat = data_concat[batch_size:]

        geo_data = dict(zip(field_key, result))
        geo_data.update(fail_field_dict)
        assert len(geo_data) == origin_size
        return json.dumps(geo_data)



if __name__ == '__main__':
    # Load model into memory
    epoch = 3000
    vae_ref = r'd:\Documents\aisr\GeosRelate\dataset_style3_slim\VolRatio\35_70\arch_GZ\27.jpg'
    opt_pickle = './checkpoints/remote/arch_layout_ngf32_volrate/opt.pkl'
    opt = EvalOptions(opt_pickle).parse()
    opt.num_upsampling_layers = 'normal'
    opt.continue_train = False
    opt.which_epoch = epoch
    opt.checkpoints_dir = './checkpoints/remote'
    print(opt)

    # vae_ref
    processor = DataPrep(vae_ref)

    # create model
    model = Pix2PixModel(opt)
    # model = UNetModel(opt)
    model.cuda()
    model.eval()

    app.run(port=8088)

    # # process an field image
    # field_files = glob.glob(r'd:\Documents\aisr\GeosRelate\field_ZS\*')
    # field_file = np.random.choice(field_files, 1)[0]
    # input_dict = processor.preprocess_image(field_file)
    #
    # # forward process
    # label = input_dict['label'].long()
    # _, h, w = label.size()
    # input_semantics = torch.FloatTensor(2, h, w).zero_()
    # input_semantics = input_semantics.scatter_(0, label, 1.0)
    # input_semantics = input_semantics.unsqueeze(0).cuda()
    # vr = input_dict['vr'].unsqueeze(0).cuda()
    # reference_image = input_dict['image'].unsqueeze(0).cuda()
    # with torch.no_grad():
    #     gen_image, _ = model.generate_fake(input_semantics, reference_image, volume_ratio=vr)
    #
    # # image postprocess
    # gen_image = tensor2im(gen_image[0])
    # save_image(gen_image, './gen.png')
