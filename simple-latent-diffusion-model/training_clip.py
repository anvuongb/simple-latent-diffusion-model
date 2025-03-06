from clip.models.clip import CLIP
from clip.models.ko_clip import KoCLIPWrapper
from helper.trainer import Trainer
from torch.utils.data import DataLoader, Dataset, TensorDataset
#from helper.data_generator import DataGenerator
import torch

class ImageTextDataset(Dataset):
    def __init__(self, image, text_prompts):
        """
        Args:
            image_paths: List of paths to images.
            text_prompts: List of corresponding text prompts.  MUST be the same length as image_paths.
            processor: The CLIP processor.
            tokenizer: The CLIP tokenizer
        """
        

        self.image = image
        self.text_prompts = text_prompts

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image = self.image[idx]
        text = self.text_prompts[idx]

        # NO!  Do *not* process here. Return raw image and text.  Processing happens in the training loop
        # to handle batching correctly with padding/truncation.
        return image, text

if __name__ == "__main__":
    #clip = CLIP('./configs/composite_clip_config.yaml')
    clip = KoCLIPWrapper()
    #dg = DataGenerator()
    num_samples = 10
    images = torch.randn(10, 3, 32, 32)
    text_prompts = [f"A photo of dummy object {i}" for i in range(num_samples)]
    #text_prompts = clip.text_encode(text_prompts, tokenize=True)
    print(text_prompts)
    data_loader = DataLoader( ImageTextDataset(images, text_prompts) )
    #dataloader = dg.composite('./datasets/image/', './datasets/json/')
    optimizer = torch.optim.AdamW([clip.logit_scale], lr=1e-4)
    trainer = Trainer(clip, clip.loss, optimizer = optimizer)
    trainer.train(data_loader, 100, 'composite_clip')