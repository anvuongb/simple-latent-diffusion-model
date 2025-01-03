import torch
from auto_encoder.models.variational_auto_encoder import VariationalAutoEncoder
import os

from helper.data_generator import DataGenerator
from helper.painter import Painter
from helper.trainer import Trainer
from helper.loader import Loader
from diffusion_model.models.latent_diffusion_model import LatentDiffusionModel
from diffusion_model.network.uncond_u_net import UnconditionalUnetwork
from diffusion_model.sampler.ddim import DDIM

IMAGE_SHAPE = (3, 32, 32)
CONFIG_PATH = './configs/cifar10_config.yaml'
VAE_FILE_NAME = './auto_encoder/check_points/vae'
DM_FILE_NAME = './diffusion_model/check_points/ldm'

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using device : {device}\t'  + (f'{torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'CPU' ))
    
    data_generator = DataGenerator()
    data_loader = data_generator.cifar10(batch_size = 128)
    painter = Painter()
    loader = Loader()

    vae = VariationalAutoEncoder(CONFIG_PATH)
    #loader.model_load('./auto_encoder/check_points/vae_epoch336', vae, ema=True)
    dat = next(iter(data_loader))[0][0:4]
    painter.show_images(dat)
    dat = vae(dat)[0]
    painter.show_images(dat)
    
    sampler = DDIM(CONFIG_PATH)
    network = UnconditionalUnetwork(CONFIG_PATH)
    dm = LatentDiffusionModel(network, sampler, vae, IMAGE_SHAPE)
    #loader.model_load('./diffusion_model/check_points/ldm_epoch300', dm, ema=True)
    
    sample = dm.generate_sequence(4)
    painter.make_gif(sample, file_name='aa')
    
    