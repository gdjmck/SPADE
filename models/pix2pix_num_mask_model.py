import torch
from models.pix2pix_condition_model import Pix2PixConditionModel

class Pix2PixNumMaskModel(Pix2PixConditionModel):
    def preprocess_input(self, data):
        # move to GPU and change data types
        data['label'] = data['label'].long()
        # data['label'][data['label'] != 1] = 0
        data['masked_image'] = data['masked_image'][:, :1, ...]
        if self.use_gpu():
            data['label'] = data['label'].cuda()
            data['image'] = data['image'].cuda()
            data['masked_image'] = data['masked_image'].cuda()

        # create one-hot label map
        label_map = data['label']
        bs, _, h, w = label_map.size()
        image_masked = data['masked_image']
        input_semantics = torch.cat([label_map, image_masked], dim=1)

        # concatenate instance map if it exists
        if not self.opt.no_instance:
            inst_map = data['instance']
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)

        condition = data.get('condition', None)
        if condition is not None:
            condition = condition.to(input_semantics.device)

        return input_semantics, data['image'], condition

    def forward(self, data, mode):
        input_semantics, real_image, condition = self.preprocess_input(data)  # vr = None if not opt.volume_rate

        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(
                input_semantics, real_image, condition)
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                input_semantics, real_image, condition)
            return d_loss
        elif mode == 'encode_only':
            z, mu, logvar = self.encode_z(real_image)
            return mu, logvar
        elif mode == 'inference':
            with torch.no_grad():
                fake_image, _, _ = self.generate_fake(input_semantics, None, condition=condition)
            return fake_image
        else:
            raise ValueError("|mode| is invalid")
