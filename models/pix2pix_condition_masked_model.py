import torch
from models.pix2pix_condition_model import Pix2PixConditionModel
from data.arch_dataset import random_mask


class Pix2PixConditionMaskedModel(Pix2PixConditionModel):
    def preprocess_input(self, data):
        # move to GPU and change data types
        data['label'] = data['label'].long()
        # data['label'][data['label'] != 1] = 0
        if self.use_gpu():
            data['label'] = data['label'].cuda()
            data['instance'] = data['instance'].cuda()
            data['image'] = data['image'].cuda()
            data['mask'] = data['mask'].cuda()
            data['image_masked'] = data['image_masked'].cuda()

        # create one-hot label map
        label_map = data['label']
        bs, _, h, w = label_map.size()
        input_semantics = torch.cat([label_map, data['mask'], data['image_masked']], dim=1)

        # concatenate instance map if it exists
        if not self.opt.no_instance:
            inst_map = data['instance']
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)

        condition = data.get('condition', None)
        if condition is not None:
            condition = condition.to(input_semantics.device)

        return input_semantics, data['image'], condition
