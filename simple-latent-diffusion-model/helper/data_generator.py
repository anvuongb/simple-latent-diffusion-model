from torchvision.datasets import CIFAR10, CelebA
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Lambda, CenterCrop, Resize
import os
import json
from PIL import Image as im
from helper.tokenizer import Tokenizer

class UnlabelDataset(Dataset):
    def __init__(self, path, transform):
        self.path = path
        self.file_list = os.listdir(path)
        self.transform = transform
        
    def __len__(self) :
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.path + self.file_list[index]
        image = im.open(img_path)
        image = self.transform(image)
        return image
    
class CompositeDataset(Dataset):
    def __init__(self, path, text_path):
        self.path = path
        self.text_path = text_path
        self.tokenizer = Tokenizer()
        
        self.file_numbers = os.listdir(path)
        self.file_numbers = [ os.path.splitext(filename)[0] for filename in self.file_numbers ]
        
        self.transform = Compose([
                ToTensor(),
                CenterCrop(400),
                Resize(256, antialias=None),
                Lambda(lambda x: (x - 0.5) * 2)
            ])
        
    def __len__(self) :
        return len(self.file_numbers)
    
    def get_text(self, text_path):
        with open(text_path, encoding = 'CP949') as f:
            text = json.load(f)['description']['impression']['description']
        return text

    def __getitem__(self, idx) :
        img_path = self.path + self.file_numbers[idx] + '.png'
        text_path = self.text_path + self.file_numbers[idx] + '.json'
        image = self.transform(im.open(img_path))
        text = self.get_text(text_path)
        text = self.tokenizer.tokenize(text)
        text = text.squeeze(0)
        return image, text

class DataGenerator():
    def __init__(self, ):
        self.transform = Compose([
            ToTensor(),
            Lambda(lambda x: (x - 0.5) * 2)
            ])
        
    def cifar10(self, path = './datasets', batch_size : int = 64, train : bool = True):
        train_data = CIFAR10(path, download = True, train = train, transform = self.transform)
        dl = DataLoader(train_data, batch_size, shuffle = True)
        return dl
    
    def celeba(self, path = './datasets', batch_size : int = 16):
        train_data = CelebA(path, transform = Compose([
            ToTensor(),
            CenterCrop(178),
            Resize(128),
            Lambda(lambda x: (x - 0.5) * 2)
            ]))
        dl = DataLoader(train_data, batch_size, shuffle = True)
        return dl
    
    def composite(self, path, text_path, batch_size : int = 16):
        return DataLoader(CompositeDataset(path, text_path), batch_size=batch_size)
        