import torch.nn as nn
import torch

from diffusion_model.network.u_net import Unet
import yaml

class UnconditionalUnetwork(nn.Module):
    def __init__(self, config_path):
        super().__init__()
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)['unet']
        self.add_module('network', Unet(dim=config['dim'], dim_mults=config['dim_mults'], channels=config['channels']))
        
    def forward(self, x, t):
        if t.dim() == 0:
            t = x.new_full((x.size(0), ), t, dtype = torch.int, device = x.device)
        return self.network(x, t)