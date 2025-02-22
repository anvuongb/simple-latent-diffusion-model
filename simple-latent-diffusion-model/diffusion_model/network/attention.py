import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = nn.RMSNorm(dim)
        self.attn = nn.MultiheadAttention(
            
            )

    def forward(self, x):
        b, _, h, w = x.shape
        
