from .test_options import TestOptions
from .train_options import TrainOptions

class ValidateOptions(TrainOptions):
    def initialize(self, parser):
        parser = super(ValidateOptions, self).initialize(parser)
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--how_many', type=int, default=float("inf"), help='how many test images to run')
        parser.add_argument('--web_ip', type=str, default='localhost', help='ip to run service app on')
        parser.add_argument('--port', type=int, default=7001, help='port to use')

        parser.set_defaults(preprocess_mode='scale_width', crop_size=256, load_size=256, display_winsize=256)
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(no_flip=True)
        parser.set_defaults(phase='test')
        self.isTrain = False
        self.initialized = True
        self.isValidate = True
        return parser
