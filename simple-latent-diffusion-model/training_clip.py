from clip.models.clip import CLIP
from clip.models.ko_clip import KoCLIPWrapper
from helper.trainer import Trainer
from torch.utils.data import DataLoader, Dataset, TensorDataset
from helper.data_generator import DataGenerator

if __name__ == "__main__":
    #clip = CLIP('./configs/composite_clip_config.yaml')
    clip = KoCLIPWrapper()
    dg = DataGenerator()
    num_samples = 10
    images = torch.randn(10, 3, 32, 32)
    text_prompts = [f"A photo of dummy object {i}" for i in range(num_samples)]
    data_loader = DataLoader(TensorDataset(images, text_prompts))
    #dataloader = dg.composite('./datasets/image/', './datasets/json/')
    trainer = Trainer(clip, clip.loss)
    trainer.train(dataloader, 100, 'composite_clip')