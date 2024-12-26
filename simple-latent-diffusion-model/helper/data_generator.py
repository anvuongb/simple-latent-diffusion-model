from torchvision.datasets import CIFAR10, CelebA
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda, CenterCrop, Resize

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