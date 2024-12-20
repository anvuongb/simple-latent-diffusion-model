from diffusion_model.network.u_net import Unet
import torch.nn as nn
import torch

class ConditionalUnetwork(nn.Module):
    def __init__(self, num_c, class_emb_size, in_channels, channels, channels_multipliers = (1, 2, 4, 8)):
        super().__init__()
        self.class_emb_size = class_emb_size
        self.embedding = nn.Embedding(num_c, self.class_emb_size)
        self.add_module('network', Unet(dim = channels, dim_mults = channels_multipliers, channels = in_channels + self.class_emb_size, out_dim = in_channels))
        
    def forward(self, x, t, cond):
        if t.dim() == 0:
            t = x.new_full((x.size(0), ), t, dtype = torch.int, device = x.device)
        n, _, w, h = x.shape
        
        cond = self.embedding(cond)
        cond = cond.view(x.size(0), self.class_emb_size, 1, 1).expand(n, self.class_emb_size, w, h)
        x = torch.cat((x, cond), 1)
        
        return self.network(x, t)