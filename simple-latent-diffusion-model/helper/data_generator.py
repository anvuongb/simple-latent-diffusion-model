from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda

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

    