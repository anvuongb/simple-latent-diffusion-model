from clip.models.clip import CLIP
from helper.trainer import Trainer
from helper.data_generator import DataGenerator

if __name__ == "__main__":
    clip = CLIP('./configs/composite_clip_config.yaml')
    dg = DataGenerator()
    dataloader = dg.composite('./datasets/image/', './datasets/json/')
    trainer = Trainer(clip, clip.loss)
    trainer.train(dataloader, 100, 'composite_clip')