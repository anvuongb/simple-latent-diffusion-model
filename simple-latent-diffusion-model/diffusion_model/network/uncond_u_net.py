from diffusion_model.network.u_net import Unet
import torch.nn as nn
import torch

class UnconditionalUnetwork(nn.Module):
    def __init__(self, in_channels, channels, channels_multipliers = (1, 2, 4, 8)):
        super().__init__()
        self.add_module('network', Unet(dim = channels, dim_mults = channels_multipliers, channels = in_channels))
        
    def forward(self, x, t):
        if t.dim() == 0:
            t = x.new_full((x.size(0), ), t, dtype = torch.int, device = x.device)
        return self.network(x, t)