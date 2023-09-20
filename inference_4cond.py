import web
import json
import torch
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.affinity import translate
from inference import GenerateLayout, DataPrep
from options.test_options import TestOptions
from options.validate_options import ValidateOptions
from models import create_model
from util.volume_rate import Condition
from util.util import tensor2im, save_image
from util.postprocess import to_geojson
from util.geo_util import longtitude_to_coord, coord_to_longtitude

try:
    opt = TestOptions().parse()
except:
    opt = ValidateOptions().parse()
    opt.isTrain = False
    opt.isValidate = True
# opt = ValidateOptions().parse()
# opt.isTrain = False
# opt.isValidate = True

model = create_model(opt)
model.eval()
pre_condition = Condition(opt)
processor = DataPrep()

class WebApp(web.application):
    def run(self, port=opt.port, *middleware):
        func = self.wsgifunc(*middleware)
        return web.httpserver.runsimple(func, (opt.web_ip, port))

urls = ('/generate', 'CondGenerateLayout')
app = WebApp(urls, globals())

condition_keymap = {'n': 'buildNum', 'd': 'density', 'v': 'volRat', 'f': 'avgFloors'}

class CondGenerateLayout(GenerateLayout):
    def preprocess_list(self, data):
        """
        :param data: zip object of (field_name, field_data)
        :return: {field_name: field_data}
        """
        field_data = {}
        for field_key, field_outloop in data:
            field_outloop = np.array(field_outloop).reshape(-1, 2).tolist()
            # 转坐标
            coords = [longtitude_to_coord(*pt) for pt in field_outloop]
            field_data[field_key] = coords
        return field_data

    def POST(self):
        web.header('Access-Control-Allow-Origin', '*')
        try:
            data = web.input()
            # 条件字典
            condition_dict = {'buildNum': json.loads(data['buildNum']),
                              'volRat': json.loads(data['volRat']),
                              'density': json.loads(data['density']),
                              'avgFloors': json.loads(data['avgFloors'])}
            # 条件准备
            # condition_names = pre_condition.condition_name[pre_condition.condition_mask]  # 模型能处理的条件
            condition_names = [condition_keymap[letter] for letter in opt.condition_order[:opt.condition_size]]
            condition_origin_vals = np.array([condition_dict[k] for k in condition_names]).astype(float).transpose(1, 0)  # 已经排好序
            condition_input = (condition_origin_vals - pre_condition.condition_mean[None, ...]) / pre_condition.condition_stdvar[None, ...]
            condition_tensor = torch.tensor(condition_input, dtype=torch.float32)
            if model.use_gpu():
                condition_tensor = condition_tensor.cuda()

            names = json.loads(data['fieldName'])
            data = json.loads(data['redLine'])
            assert len(data) == len(names)
        except KeyError:
             return {'code': 400, 'msg': '数据格式有误'}
        if len(data) == 0:
            return []
        origin_size = len(data)
        data = self.preprocess_list(zip(names, data))

        input_dict_list = []
        real_centers = []  # 真实投影坐标的中心点
        field_size = []
        field_key = []
        field_names = []
        fail_field_dict = {}
        keep_condition_index = list(range(condition_tensor.size(0)))
        for i, field_name in enumerate(data.keys()):
            field_poly = Polygon(np.array(data[field_name]).reshape(-1, 2))
            bias = self.get_bias(field_poly)
            real_centers.append([field_poly.centroid.x, field_poly.centroid.y])
            field_size.append((field_poly.bounds[2]-field_poly.bounds[0],
                               field_poly.bounds[3]-field_poly.bounds[1]))
            field_names.append(field_name)
            print('入参图形的重心:{}'.format(real_centers[-1]))
            field_poly = translate(field_poly, **bias)
            self.image_saver.save(field_name, gpd.GeoSeries(field_poly))
            try:
                input_dict_list.append(processor.preprocess_image(self.get_file(field_name)))
            except:
                # 舍弃当前数据
                keep_condition_index.remove(i)
                field_names.pop()
                field_size.pop()
                real_centers.pop()
                # 记录无法处理的数据
                fail_field_dict[field_name] = "无法处理地块"
                continue

            field_key.append(field_name)
        condition_tensor = condition_tensor[keep_condition_index]
        num_fields = len(field_key)

        # 构造batch data
        input_semantics_list = []
        for input_dict in input_dict_list:
            input_semantics_list.append(input_dict['label'].float().unsqueeze(0))
        if len(input_semantics_list) == 0:
            return []
        # 数据拼接
        input_semantics_tensor = torch.cat(input_semantics_list, 0).cuda()

        # forward
        with torch.no_grad():
            b, c, h, w = input_semantics_tensor.size()
            gen_image, _, _ = model.generate_fake(input_semantics_tensor, None, condition=condition_tensor)
            _, condition_pred, _, _ = model.discriminate(input_semantics_tensor, gen_image, gen_image)
            for bid in range(b):
                condition_read = pre_condition.read_condition(condition_tensor[bid: bid+1].cpu().numpy())
                condition_pred_i = pre_condition.read_condition(condition_pred[bid: bid+1].cpu().numpy())
                print('read: {}\tpred: {}'.format(condition_read, condition_pred_i))
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
            output['name'] = field_names[i]
            # 转换为geojson
            output = to_geojson(output)
            # 恢复偏移
            result.append(output)

        geo_data = dict(zip(field_key, result))
        geo_data.update(fail_field_dict)
        assert len(geo_data) == origin_size
        return json.dumps(geo_data)


if __name__ == '__main__':
    app.run()