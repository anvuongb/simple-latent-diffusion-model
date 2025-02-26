import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, emb_dim):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.gamma_proj = nn.Linear(emb_dim, num_channels)
        self.beta_proj = nn.Linear(emb_dim, num_channels)
        self.initialize()

    def initialize(self):
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.ones_(self.gamma_proj.bias)  # Gamma starts as 1
        nn.init.zeros_(self.beta_proj.bias)  # Beta starts as 0

    def forward(self, x, emb):
        # [B, C, H, W] -> [B, C, 1, 1] for broadcasting
        gamma = self.gamma_proj(emb).view(x.size(0), -1, 1, 1)
        beta = self.beta_proj(emb).view(x.size(0), -1, 1, 1)
        return gamma * self.norm(x) + beta


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups, time_emb_dim, cond_emb_dim=None, dropout=0.0, use_depthwise=False):
        super().__init__()
        self.norm_time = AdaptiveGroupNorm(groups, dim, time_emb_dim)
        self.use_cond = cond_emb_dim is not None
        if self.use_cond:
            self.norm_cond = AdaptiveGroupNorm(groups, dim, cond_emb_dim)

        if use_depthwise:
            self.proj = DepthwiseSeparableConv2d(dim, dim_out, kernel_size=3, padding=1)
        else:
            self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, time_emb, cond_emb=None):
        x = self.norm_time(x, time_emb)  # Timestep AdaGN
        if self.use_cond:
            x = self.norm_cond(x, cond_emb) # Condition AdaGN

        x = self.proj(x)
        x = self.act(x)
        return self.dropout(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim: int, dim_out: int, t_emb_dim: int, *,
                 y_emb_dim: int = None, groups: int = 32, dropout: float = 0.0, residual_scale=1.0):
        super().__init__()
        self.block1 = Block(dim, dim_out, groups, t_emb_dim, y_emb_dim, dropout)
        self.block2 = Block(dim_out, dim_out, groups, t_emb_dim, y_emb_dim, dropout)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.residual_scale = nn.Parameter(torch.tensor(residual_scale))


    def forward(self, x, t_emb, y_emb=None):
        h = self.block1(x, t_emb, y_emb)
        h = self.block2(h, t_emb, y_emb)
        return self.residual_scale * h + self.res_conv(x)

class DepthwiseSeparableConv2d(nn.Module): # Added
    def __init__(self, dim_in, dim_out, kernel_size, padding):
        super().__init__()
        self.depthwise = nn.Conv2d(dim_in, dim_in, kernel_size, padding=padding, groups=dim_in)
        self.pointwise = nn.Conv2d(dim_in, dim_out, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x