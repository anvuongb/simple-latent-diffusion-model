# Welcome to the Simple Latent Diffusion Model

이 저장소는 간단한 잠재 확산 모형(Latent Diffusion Model)의 구현을 제공합니다. 코드와 내용은 지속적으로 갱신될 예정입니다.

| 데이터 세트                                | 잠재 변수의 생성 프로세스               | 생성된 데이터                          |
|---------------------------------------------|-----------------------------------------|-----------------------------------------|
| Swiss-roll  | <img src="assets/swiss_roll.gif" width="300"/>   | <img src="assets/swiss_roll_image.png" width="300"/>  |
| CIFAR-10  | <img src="assets/cifar10.gif" width="300"/>   | <img src="assets/cifar10_image.png" width="300"/>  |
| CelebA  | <img src="assets/celeba.gif" width="300"/>   | <img src="assets/celeba_image.png" width="300"/>  |

## 튜토리얼

- [Tutorial for Latent Diffusion Model](notebook/simple_latent_diffusion_model_tutorial.ipynb)

## 사용법

다음 예시는 코드를 이용하여 어떻게 모델을 훈련하고 데이터를 생성할 수 있는지 보여줍니다.

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

## 참고
- [lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)
- [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion)
- [labmlai/annotated_deep_learning_paper_implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations/tree/master/labml_nn/diffusion/stable_diffusion)
