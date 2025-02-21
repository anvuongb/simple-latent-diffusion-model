import torch
import torch.nn as nn

class BaseUnet(nn.Module):
    def __init__(
        self, 
        dim,
        dim_mults = [1, 2, 4, 8],
        channels : int = 3):
        super().__init__()

        self.channels = channels
        self.dim = dim
        
        self.init_conv = nn.Conv2d(self.channels, self.dim, 7, padding = 3)

        dims = [self.dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))