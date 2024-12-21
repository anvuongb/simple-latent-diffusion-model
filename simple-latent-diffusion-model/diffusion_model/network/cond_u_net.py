import torch.nn as nn
import torch

from diffusion_model.network.u_net import Unet
import yaml

class ConditionalUnetwork(nn.Module):
    def __init__(self, config_path):
        super().__init__()
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)['unet']
        
        self.class_emb_size = config['class_emb_size']
        self.embedding = nn.Embedding(config['num_class'], self.class_emb_size)
        self.add_module('network', Unet(dim=config['dim'], dim_mults=config['dim_mults'], channels=config['channels']+self.class_emb_size, out_dim=config['channels']))
        
    def forward(self, x, t, cond):
        if t.dim() == 0:
            t = x.new_full((x.size(0), ), t, dtype = torch.int, device = x.device)
        n, _, w, h = x.shape
        cond = self.embedding(cond)
        cond = cond.view(x.size(0), self.class_emb_size, 1, 1).expand(n, self.class_emb_size, w, h)
        x = torch.cat((x, cond), 1)
        return self.network(x, t)