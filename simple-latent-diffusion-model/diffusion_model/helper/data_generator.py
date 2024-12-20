from torchvision.transforms import Compose, ToTensor, Lambda
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader

class DataGenerator():
    def __init__(self):
        pass
    
    def image_transform(self):
        transform = Compose([
            ToTensor(),
            Lambda(lambda x: (x - 0.5) * 2)
            ])
        return transform
    
    def fashion_mnist(self, path = "datasets/", batch_size : int = 64, transform = None,
                      train : bool = True):
        if transform is None: transform = self.image_transform()
        train_data = FashionMNIST(path, train = train, transform = transform)
        data_loader = DataLoader(train_data, batch_size)
        return data_loader