import torch
import torch.nn as nn
import yaml

from clip.models.clip import CLIP

class CLIPEncoder(nn.Module):
    def __init__(
        self,
        clip: CLIP
        ):
        super().__init__()
        self.clip = clip
        self.clip.eval()
        for param in self.clip.parameters():
            param.requires_grad = False

    def forward(self, y):
        if isinstance(y, str):
            return self.clip.text_encode(y, tokenize=True)
        else:
            return self.clip.text_encode(y, tokenize=False)

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
        cond_dim = config['cond_dim']
        
        if cond_type == 'class':
            self.embed = nn.Embedding(num_cond, embed_dim)
        elif cond_type == 'numeric':
            self.embed = nn.Linear(num_cond, embed_dim)

        self.cond_mlp = nn.Sequential(
            self.embed,
            nn.Linear(embed_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim)
            )

    def forward(self, y: torch.tensor):
        return self.cond_mlp(y)