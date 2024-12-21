import torch
from auto_encoder.models.variational_auto_encoder import VariationalAutoEncoder
import os

from helper.data_generator import DataGenerator
from helper.painter import Painter
from helper.trainer import Trainer
from helper.loader import Loader
from diffusion_model.models.latent_diffusion_model import LatentDiffusionModel
from diffusion_model.network.cond_u_net import ConditionalUnetwork
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

    vae = VariationalAutoEncoder(CONFIG_PATH)
    #trainer = Trainer(vae, loss_fn = vae.loss)
    #trainer.train(data_loader, 300, VAE_FILE_NAME, True)
    
    sampler = DDIM(CONFIG_PATH)
    network = ConditionalUnetwork(CONFIG_PATH)
    dm = LatentDiffusionModel(network, sampler, vae, IMAGE_SHAPE)
    trainer = Trainer(dm, dm.loss)
    trainer.train(data_loader, 300, DM_FILE_NAME, False)
    
    