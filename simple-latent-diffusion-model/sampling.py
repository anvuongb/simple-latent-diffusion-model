import torch
#from auto_encoder.models.variational_auto_encoder import VariationalAutoEncoder
import os
#from clip.models.clip import CLIP
#from diffusion_model.models.clip_latent_diffusion_model import CLIPLatentDiffusionModel
#from diffusion_model.network.cond_u_net import ConditionalUnetwork

#from diffusion_model.models.uncond_diffusion_model import UnconditionalDiffusionModel
from diffusion_model.models.diffusion_model import DiffusionModel
from diffusion_model.network.base_unet import *
from helper.data_generator import DataGenerator
from helper.painter import Painter
#from diffusion_model.models.latent_diffusion_model import LatentDiffusionModel
#from diffusion_model.network.uncond_u_net import UnconditionalUnetworkWrapper
from diffusion_model.sampler.ddim import DDIM

IMAGE_SHAPE = (3, 32, 32)
CONFIG_PATH = './configs/cifar10_config.yaml'

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f'using device : {device}\t' + (f'{torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'CPU' ))
    
    data_generator = DataGenerator()
    data_loader = data_generator.random_data(size = (16, 3, 32, 32))
    painter = Painter()
    
    sampler = DDIM(CONFIG_PATH)
    network = BaseUnet(CONFIG_PATH)
    dm = DiffusionModel(UnetWrapper(BaseUnet, config_path), sampler, IMAGE_SHAPE)
    
    sample = dm()
    painter.show_images(sample)
    
    