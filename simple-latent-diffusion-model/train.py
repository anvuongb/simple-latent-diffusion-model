import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch

from helper.painter import Painter
from helper.trainer import Trainer
from helper.data_generator import DataGenerator
from helper.loader import Loader
from helper.cond_encoder import CLIPEncoder

from auto_encoder.models.variational_auto_encoder import VariationalAutoEncoder
from clip.models.ko_clip import KoCLIPWrapper
from diffusion_model.sampler.ddim import DDIM
from diffusion_model.models.latent_diffusion_model import LatentDiffusionModel
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

# Train the Variational Autoencoder (VAE)
data_loader = data_generator.cifar10(batch_size=512)
vae = VariationalAutoEncoder(CONFIG_PATH)  # Initialize the VAE model
trainer = Trainer(vae, vae.loss)  # Create a trainer for the VAE
trainer.train(dl=data_loader, epochs=400, file_name='vae', no_label=True)  # Train the VAE

# Train the Latent Diffusion Model (LDM)
data_loader = data_generator.cifar10(batch_size=128)
sampler = DDIM(CONFIG_PATH)  # Initialize the DDIM sampler
network = UnetWrapper(Unet, CONFIG_PATH, None)  # Initialize the U-Net network
ldm = LatentDiffusionModel(network, sampler, vae)  # Initialize the LDM
trainer = Trainer(ldm, ldm.loss)  # Create a trainer for the LDM
trainer.train(dl=data_loader, epochs=400, file_name='ldm', no_label=True)
# Train the LDM; set 'no_label=True' if the dataset does not include labels

# Load the trained models
vae = loader.model_load('models/vae', vae, is_ema=True)
ldm = loader.model_load('models/ldm', ldm, is_ema=True)

# Generate samples using the trained latent diffusion model
ldm.eval()
ldm = ldm.to(device)
sample = ldm(n_samples=4, y = '...', gamma = 3)  # Generate 4 sample images, 'y' represents any conditions, 'gamma' means guidance scale
painter.show_images(sample)  # Display the generated images