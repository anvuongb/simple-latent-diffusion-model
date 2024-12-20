import torch
import torch.nn as nn

from auto_encoder.components.normalize import Normalize

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels : int, out_channels : int = None, dropout):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        
        self.nonlinearity = nn.SiLU()
        
        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)

        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
        
        if self.in_channels != self.out_channels:
            self.shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0)
            
    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = self.nonlinearity(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        if self.in_channels != self.out_channels:
            x = self.shortcut(x)
            
        return x + h
        
        