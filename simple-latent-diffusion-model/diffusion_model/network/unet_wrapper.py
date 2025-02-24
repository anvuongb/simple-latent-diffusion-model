import torch
import torch.nn as nn
import yaml

class UnetWrapper(nn.Module):
    def __init__(self, Unet : nn.Module, config_path):
        super().__init__()
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)['unet']
        self.add_module('network', Unet(**config))
        
    def forward(self, x, t, y=None):
        if t.dim() == 0:
            t = x.new_full((x.size(0), ), t, dtype = torch.int, device = x.device)
        if y is not None:
            y = y.unsqueeze(1)
            return self.network(x, t, y)
        else: 
            return self.network(x, t)