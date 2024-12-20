import torch
import os
from auto_encoder.helper.data_generator import DataGenerator
from auto_encoder.models.variational_auto_encoder import VariationalAutoEncoder

from auto_encoder.helper.data_generator import DataGenerator
from auto_encoder.helper.loader import Loader
from diffusion_model.helper.loader import Loader2
from diffusion_model.helper.painter import Painter
from diffusion_model.models.cond_diffusion_model import ConditionalDiffusionModel
from diffusion_model.models.latent_diffusion_model import LatentDiffusionModel
from diffusion_model.network.cond_u_net import ConditionalUnetwork
from diffusion_model.sampler.ddpm import DDPM
from diffusion_model.sampler.ddim import DDIM
from diffusion_model.network.uncond_u_net import UnconditionalUnetwork
from diffusion_model.models.uncond_diffusion_model import UnconditionalDiffusionModel
from diffusion_model.helper.trainer import Trainer
from diffusion_model.helper.beta_generator import BetaGenerator

IMAGE_SHAPE = (3, 32, 32)

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using device : {device}\t'  + (f'{torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'CPU' ))
    dg = DataGenerator()
    dl = dg.cifar10(batch_size = 128)
    
    ae = VariationalAutoEncoder(config_path = './auto_encoder/configs/cifar10_config.yaml',embed_dim=8).to(device)
    
    ld = Loader()
    ld.load_with_acc(file_name = './auto_encoder/check_points/cifar10_epoch81',model = ae)
    
    #sampler = DDPM(1000, BetaGenerator(1000).linear_beta_schedule())
    sampler = DDIM(1000, BetaGenerator(1000).linear_beta_schedule(), ddim_T = 20)
    network = ConditionalUnetwork(10, 4, 8, 32, (1, 2, 4))
    model = LatentDiffusionModel(network, sampler, ae, IMAGE_SHAPE).to(device)
    trainer = Trainer(model, model.loss, device = device)
    loader = Loader()
    pt = Painter()
    dg = DataGenerator()
    dl = dg.cifar10()
    a = model(cond = 1)
    pt.show_images(a)
    loader = Loader2()
    #trainer.accelerated_train(dl, epochs = 100, file_name = 'diffusion_model/check_points/ldm', no_label = False)
    loader.load('diffusion_model/check_points/ldm_epoch57',model, ema = True, print_dict=True)
    a = model(1)
    print(a)
    pt.show_images(a)
    
    
    
    

    
    
    
   
   