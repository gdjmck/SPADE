from models.pix2pix_condition_model import Pix2PixConditionModel

class StyleConditionModel(Pix2PixConditionModel):
    def generate_fake(self, input_semantics, real_image, compute_kld_loss=False, condition=None):
        assert condition is not None
        fake_image = self.netG(input_semantics, z=condition)
        return fake_image, None, None