import torch
import torch.nn as nn

class ConditionEncoder(nn.Module):
    def __init__(
        self, 
        cond_type : str,
        num_cond : int,
        embed_dim: int = 128,
        ):
        super().__init__()
        if cond_type == 'class':
            self.embed = nn.Embedding(num_cond, embed_dim)
        elif cond_type == 'numeric':
            self.embed = nn.Linear(num_cond, embed_dim)

        self.cond_mlp = nn.Sequential(
            self.embed,
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
            )

    def forward(self, y: torch.tensor):
        return self.cond_mlp(y)