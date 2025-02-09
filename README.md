# Welcome to the Simple Latent Diffusion Model

🌐 README in Korean: [KR 한국어 버전](README_ko.md)

This repository contains a simplified implementation of a latent diffusion model. The code and contents will be updated continuously.
| **Dataset**                                     | **Generation Process of Latents**           | **Generated Data**                          |
|---------------------------------------------|-----------------------------------------|-----------------------------------------|
| Swiss-roll  | <img src="assets/swiss_roll.gif" width="300"/>   | <img src="assets/swiss_roll_image.png" width="300"/>  |
| CIFAR-10  | <img src="assets/cifar10.gif" width="300"/>   | <img src="assets/cifar10_image.png" width="300"/>  |
| CelebA  | <img src="assets/celeba.gif" width="300"/>   | <img src="assets/celeba_image.png" width="300"/>  |

## **Generate Composites with CLIP**

The table below showcases text-to-image generation using CLIP. The dataset used is the [Asian Composite Dataset](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=618), with input text in Korean.

| **English Text** | **Generated Image** |
|-----------------|------------------------------|
| A round face with voluminous, slightly long short hair, along with barely visible vocal cords, gives off a more feminine aura than a masculine one. The well-defined eyes and lips enhance the subject's delicate features, making them appear more refined and intellectual. | <img src="assets/Ex1.png" width="1000"/> |
| The hairstyle appears slightly unpolished, lacking a refined touch. The slightly upturned eyes give off a sharp and somewhat sensitive impression. Overall, they seem to have a slender physique and appear efficient in handling tasks, though their social interactions may not be particularly smooth. | <img src="assets/Ex4.png" width="1000"/> | 

### Why doesn't the generated image match the input text?

This is because the sampling was performed without guidance. Using classifier-free guidance (CFG) during sampling can significantly improve sample quality and enhance the performance of conditional generation.


## **Tutorials**

- [Tutorial for Latent Diffusion Model](notebook/simple_latent_diffusion_model_tutorial.ipynb)

## **Usage**

The following example demonstrates how to use the code in this repository.

```python
import torch
import os

from auto_encoder.models.variational_auto_encoder import VariationalAutoEncoder
from helper.data_generator import DataGenerator
from helper.painter import Painter
from helper.trainer import Trainer
from helper.loader import Loader
from diffusion_model.models.latent_diffusion_model import LatentDiffusionModel
from diffusion_model.network.uncond_u_net import UnconditionalUnetwork
from diffusion_model.sampler.ddim import DDIM

# Path to the configuration file
CONFIG_PATH = './configs/cifar10_config.yaml'

# Instantiate helper classes
painter = Painter()
loader = Loader()
data_generator = DataGenerator()

# Load CIFAR-10 dataset
data_loader = data_generator.cifar10(batch_size=128)

# Train the Variational Autoencoder (VAE)
vae = VariationalAutoEncoder(CONFIG_PATH)  # Initialize the VAE model
trainer = Trainer(vae, vae.loss)  # Create a trainer for the VAE
trainer.train(dl=data_loader, epochs=1000, file_name='vae', no_label=True)  # Train the VAE

# Train the Latent Diffusion Model (LDM)
sampler = DDIM(CONFIG_PATH)  # Initialize the DDIM sampler
network = UnconditionalUnetwork(CONFIG_PATH)  # Initialize the U-Net network
ldm = LatentDiffusionModel(network, sampler, vae, image_shape=(3, 32, 32))  # Initialize the LDM
trainer = Trainer(ldm, ldm.loss)  # Create a trainer for the LDM
trainer.train(dl=data_loader, epochs=1000, file_name='ldm', no_label=True)  
# Train the LDM; set 'no_label=False' if the dataset includes labels

# Generate samples using the trained diffusion model
ldm = LatentDiffusionModel(network, sampler, vae, image_shape=(3, 32, 32))  # Re-initialize the LDM
loader.model_load('./diffusion_model/check_points/ldm_epoch1000', ldm, ema=True)  # Load the trained model
sample = ldm(n_samples=4)  # Generate 4 sample images
painter.show_images(sample)  # Display the generated images
```


## **References**
- [lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)
- [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion)
- [labmlai/annotated_deep_learning_paper_implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations/tree/master/labml_nn/diffusion/stable_diffusion)
