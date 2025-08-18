import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch

from helper.painter import Painter
from helper.trainer import Trainer
from helper.data_generator import DataGenerator
from helper.loader import Loader
from helper.cond_encoder import CLIPEncoder

from clip.models.ko_clip import KoCLIPWrapper
from diffusion_model.sampler.ddim import DDIM
from diffusion_model.models.latent_diffusion_model import DiffusionModel
from diffusion_model.network.unet import Unet
from diffusion_model.network.unet_wrapper import UnetWrapper

# Path to the configuration file
# CONFIG_PATH = './configs/celeba_config.yaml'
CONFIG_PATH = './configs/cifar10_config.yaml'


# Set device
device = torch.device('cuda')

# Instantiate helper classes
painter = Painter()
loader = Loader()
data_generator = DataGenerator()

# Load CIFAR-10 dataset
# data_loader = data_generator.celeba(batch_size=128)


## Load CLIP model
#clip = KoCLIPWrapper() # Any CLIP model from Hugging Face
#cond_encoder = CLIPEncoder(clip, CONFIG_PATH) # Set encoder

# Train the Latent Diffusion Model (LDM)
data_loader = data_generator.cifar10(batch_size=1024)
sampler = DDIM(CONFIG_PATH)  # Initialize the DDIM sampler
network = UnetWrapper(Unet, CONFIG_PATH, None)  # Initialize the U-Net network
dm = DiffusionModel(network, sampler, image_shape=[3,32,32])  # Initialize the LDM
trainer = Trainer(dm, dm.loss)  # Create a trainer for the LDM
trainer.train(dl=data_loader, epochs=800, file_name='dm', no_label=True)
# Train the LDM; set 'no_label=True' if the dataset does not include labels
