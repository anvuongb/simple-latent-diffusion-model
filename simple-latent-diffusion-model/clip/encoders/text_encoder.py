import torch
import torch.nn as nn

class ImageEncoder(nn.Module):
    def __init__(self, in_channels: int, resolution: int, patch_size: int, 
                 number_of_features: int, number_of_heads: int, number_of_transformer_layers: int,
                 embed_dim: int):
        super().__init__()
        self.resolution = resolution
        self.embed_dim = embed_dim
        
        # Convolution for patch embedding
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=number_of_features, 
                              kernel_size=patch_size, stride=patch_size, bias=False)
        self.number_of_patches = (resolution // patch_size) ** 2

        # Positional and Class Embeddings
        self.positional_embedding = nn.Parameter(torch.zeros(1, self.number_of_patches + 1, number_of_features))
        self.class_embedding = nn.Parameter(torch.zeros(1, 1, number_of_features))
        
        # Layer Normalization and Transformer
        self.ln_pre = nn.LayerNorm(number_of_features)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=number_of_features, nhead=number_of_heads, batch_first=True),
            num_layers=number_of_transformer_layers
        )
        self.ln_post = nn.LayerNorm(number_of_features)
        self.fc = nn.Linear(number_of_features, embed_dim)
        
        # Initialize weights
        nn.init.normal_(self.positional_embedding, std=0.02)
        nn.init.normal_(self.class_embedding, std=0.02)
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity='relu')
    
    def forward(self, x: torch.Tensor):
        x = self.conv(x)  # [batch_size, number_of_features, grid, grid]
        x = x.flatten(2).transpose(1, 2)  # [batch_size, number_of_patches, number_of_features]
        
        # Add class token and positional embedding
        class_embeddings = self.class_embedding.expand(x.shape[0], -1, -1)
        x = torch.cat([class_embeddings, x], dim=1)
        x = x + self.positional_embedding
        
        # Pre-normalization, Transformer, and Post-normalization
        x = self.ln_pre(x)
        x = self.transformer(x)
        class_token = x[:, 0]  # Class token
        x = self.ln_post(class_token)
        x = self.fc(x)  # Final embedding
        
        return x
