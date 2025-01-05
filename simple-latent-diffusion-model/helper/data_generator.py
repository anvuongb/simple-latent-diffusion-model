from torchvision.datasets import CIFAR10, CelebA
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Lambda, CenterCrop, Resize
import os
from PIL import Image as im

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