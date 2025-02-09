# Welcome to the Simple Latent Diffusion Model

이 저장소는 간단한 잠재 확산 모형(Latent Diffusion Model)의 구현을 제공합니다. 코드와 내용은 지속적으로 갱신될 예정입니다.

| 데이터 세트                                | 잠재 변수의 생성 프로세스               | 생성된 데이터                          |
|---------------------------------------------|-----------------------------------------|-----------------------------------------|
| Swiss-roll  | <img src="assets/swiss_roll.gif" width="300"/>   | <img src="assets/swiss_roll_image.png" width="300"/>  |
| CIFAR-10  | <img src="assets/cifar10.gif" width="300"/>   | <img src="assets/cifar10_image.png" width="300"/>  |
| CelebA  | <img src="assets/celeba.gif" width="300"/>   | <img src="assets/celeba_image.png" width="300"/>  |

## CLIP을 이용한 text-to-image 생성

아래 테이블은 CLIP을 이용한 text-to-image 결과를 나타냅니다. 데이터 세트는 [페르소나 기반의 가상 몽타주 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=618)입니다.

| **입력 텍스트** | **생성된 이미지** |
|----------------|------------------------------|
|동그란 얼굴에 풍성하고 살짝 긴 커트머리와 거의 나오지 않은 성대가 남성보다는 여성적인 분위기를 냅니다. 또렷한 눈과 입술이 인물의 섬세함을 더욱 부각시키고 지적으로 보이게 만듭니다. | <img src="assets/Ex1.png" width="600"/> |
|헤어 손질이 다소 미숙하여 세련된 느낌이 부족하다. 눈 끝이 올라가 있어 눈빛이 날카롭고 예민해 보인다. 전체적으로 마른 체격일 것으로 보이며, 업무 처리 능력은 뛰어나겠지만, 교우 관계는 원만하지 않을 수도 있다. | <img src="assets/Ex4.png" width="600"/> | 

### 왜 입력 텍스트와 이미지가 잘 맞지 않는가?

이는 guidance 없이 데이터를 생성했기 때문입니다. Classifier-free guidance (CFG)를 이용하면 생성 데이터의 품질과 조건부 생성의 성능을 향상시킬 수 있습니다.

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
