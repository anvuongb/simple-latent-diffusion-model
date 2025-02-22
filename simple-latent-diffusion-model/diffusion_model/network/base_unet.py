# Refer: # https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
import torch
import torch.nn as nn
from torch.nn.modules import MultiheadAttention
import yaml

from diffusion_model.network.timestep_embedding import SinusoidalEmbedding

class DownSample(nn.Module):
    def __init__(self, dim: int):
        """
        Downsamples the spatial dimensions by a factor of 2 using a strided convolution.

        Args:
            dim: Input channel dimension.
        """
        super().__init__()
        self.downsample = nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            Downsampled tensor of shape [B, C, H/2, W/2].
        """
        return self.downsample(x)

class UpSample(nn.Module):
    def __init__(self, dim: int):
        """
        Upsamples the spatial dimensions by a factor of 2 using a transposed convolution.

        Args:
            dim: Input channel dimension.
        """
        super().__init__()
        self.upsample = nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            Upsampled tensor of shape [B, C, 2*H, 2*W].
        """
        return self.upsample(x)

class AdaptiveGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, embed_dim, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.norm = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=False)

        # Linear layers to project the embedding to gamma and beta
        self.gamma_proj = nn.Linear(embed_dim, num_channels)
        self.beta_proj = nn.Linear(embed_dim, num_channels)

    def forward(self, x, embed):
        normalized = self.norm(x)
        gamma = self.gamma_proj(embed)
        beta = self.beta_proj(embed)
        gamma = gamma.view(-1, self.num_channels, 1, 1)
        beta = beta.view(-1, self.num_channels, 1, 1)
        return gamma * normalized + beta

class ResNetBlock(nn.Module):
    def __init__(self, dim: int, dim_out: int, timestep_embed_dim: int, dropout_prob: float = 0.0):
        """
        A ResNet block with adaptive group normalization for timestep conditioning.

        Args:
            dim: Input channel dimension.
            dim_out: Output channel dimension.
            timestep_embed_dim: Dimension of the timestep embedding.
            num_groups: Number of groups for Group Normalization.
            dropout_prob: Dropout probability.
        """
        super().__init__()
        self.in_layers = nn.Sequential(
            AdaptiveGroupNorm(32, dim, timestep_embed_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout_prob),
            nn.Conv2d(dim, dim_out, 3, padding=1)
            )

        self.out_layers = nn.Sequential(
            AdaptiveGroupNorm(32, dim, timestep_embed_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout_prob),
            nn.Conv2d(dim_out, dim_out, 3, padding=1)
            )

        # Zero-initialize the last convolution
        nn.init.zeros_(self.out_layers[-1].weight)
        nn.init.zeros_(self.out_layers[-1].bias)

        if dim == dim_out:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(dim, dim_out, 1)

    def forward(self, x : torch.tensor, timestep_embed : torch.tensor):
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, C_in, H, W].
            timestep_embed: Timestep embedding tensor of shape [B, embed_dim].

        Returns:
            Output tensor of shape [B, C_out, H, W].
        """
        h = self.in_layers(x, timestep_embed) # Pass embedding to AdaGN
        h = self.out_layers(h, timestep_embed)
        return self.skip_connection(x) + h

class BaseUnet(nn.Module):
    def __init__(
        self, 
        channels,
        channel_mults = [1, 2, 4, 8],
        in_channels: int = 3,
        out_channels: int = 3,
        num_resnet_blocks: int = 2,
        num_attn_heads: int = 4,
        ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        
        self.init_conv = nn.Conv2d(self.in_channels, self.channels, 7, padding = 3)

        channels_list = [channels * m for m in channel_mults]

        # timestep embedding

        timestep_emb_dim = channels * 4
        self.time_embedding = nn.Sequential(
            SinusoidalEmbedding(channels),
            nn.Linear(channels, timestep_emb_dim),
            nn.SiLU(),
            nn.Linear(timestep_emb_dim, timestep_emb_dim)
            )

        # layers

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        input_block_channels = [channels]
        levels = len(channels_list)

        for i in range(levels):
            for _ in range(num_resnet_blocks):
                layers = nn.ModuleList([
                    ResNetBlock(channels, channels_list[i], timestep_emb_dim)
                    ])
                channels = channels_list[i]
                layers.append(MultiheadAttention(channels, num_attn_heads))
                input_block_channels.append(channels)
                if i != levels - 1:
                    layers.append(DownSample(channels))
                    input_block_channels.append(channels)
                self.downs.append(layers)

        self.mid = nn.Sequential(
            ResNetBlock(channels, channels, timestep_emb_dim),
            MultiheadAttention(channels, num_attn_heads),
            ResNetBlock(channels, channels, timestep_emb_dim),
            )

        for i in reversed(range(levels)):
            for j in range(num_resnet_blocks):
                layers = nn.ModuleList([
                    ResNetBlock(channels + input_block_channels.pop(), channels_list[i], timestep_emb_dim)
                    ])
                channels = channels_list[i]
                layers.append(MultiheadAttention(channels, num_attn_heads))
                if i != 0 and j == num_resnet_blocks:
                    layers.append(UpSample(channels))
                self.ups.append(layers)

        self.out = nn.Sequential(
            AdaptiveGroupNorm(32, channels, timestep_emb_dim),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1)
            )

    def forward(self, x: torch.tensor, t: torch.tensor):
        x = self.init_conv(x)
        h = []

        t_emb = self.time_embedding(t)
        
        for m in self.downs:
            x = m(x, t_emb)
            h.append(x)

        x = self.mid(x, t_emb)

        for m in self.ups:
            x = torch.cat([x, h.pop()], dim = 1)
            x = m(x, t_emb)

        return self.out(x)

class UnetWrapper(nn.Module):
    def __init__(self, Unet : BaseUnet, config_path):
        super().__init__()
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)['unet']
        self.add_module('network', Unet(**config))
        
    def forward(self, x, t):
        if t.dim() == 0:
            t = x.new_full((x.size(0), ), t, dtype = torch.int, device = x.device)
        return self.network(x, t)