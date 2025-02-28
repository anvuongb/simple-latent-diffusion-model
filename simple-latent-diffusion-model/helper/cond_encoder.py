import torch
import torch.nn as nn
import yaml

class ConditionEncoder(nn.Module):
    def __init__(
        self, 
        config_path
        ):
        super().__init__()
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)['cond_encoder']
        cond_type = config['cond_type']
        num_cond = config['num_cond']
        embed_dim = config['embed_dim']

        if cond_type == 'class':
            self.embed = nn.Embedding(num_cond, 64)
        elif cond_type == 'numeric':
            self.embed = nn.Linear(num_cond, embed_dim)

        self.cond_mlp = nn.Sequential(
            self.embed,
            nn.Linear(64, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
            )

    def forward(self, y: torch.tensor):
        return self.cond_mlp(y)