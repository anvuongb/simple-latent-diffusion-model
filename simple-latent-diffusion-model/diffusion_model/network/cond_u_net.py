import torch.nn as nn
import torch

from diffusion_model.network.u_net import Unet
import yaml

class CrossAttention(nn.Module):
    def __init__(self, latent_dim, text_dim):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, latent_dim)
        self.attention = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=1)

    def forward(self, latent, text_embedding):
        B, C, H, W = latent.shape  # [4, 3, 16, 16]

        latent_flattened = latent.view(B, C, H * W).permute(2, 0, 1)  # [B, C, H*W] -> [H*W, B, C]

        text_embedding_proj = self.text_proj(text_embedding)  # [B, text_dim] -> [B, latent_dim]
        text_embedding_proj = text_embedding_proj.unsqueeze(0)  # [B, latent_dim] -> [1, B, latent_dim]

        attn_output, _ = self.attention(
            query=latent_flattened,         # [H*W, B, C]
            key=text_embedding_proj,        # [1, B, C]
            value=text_embedding_proj       # [1, B, C]
        )

        attn_output = attn_output.permute(1, 2, 0).view(B, C, H, W)  # [H*W, B, C] -> [B, C, H, W]

        return latent + attn_output

class ConditionalUnetwork(nn.Module):
    def __init__(self, config_path):
        super().__init__()
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)['unet']
        self.add_module('network', Unet(dim=config['dim'], dim_mults=config['dim_mults'], channels=config['channels']))
        self.add_module('cross_attention', CrossAttention(latent_dim=config['latent_dim'], text_dim=config['text_dim']))
        
    def forward(self, x, t, y):
        if t.dim() == 0:
            t = x.new_full((x.size(0), ), t, dtype = torch.int, device = x.device)
                
        x = self.cross_attention(x, y)
        x = self.network(x, t)
        x = self.cross_attention(x, y)

        return x