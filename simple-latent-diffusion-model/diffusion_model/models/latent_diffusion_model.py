import torch
import torch.nn as nn

from auto_encoder.models.variational_auto_encoder import VariationalAutoEncoder
from diffusion_model.models.uncond_diffusion_model import UnconditionalDiffusionModel

class LatentDiffusionModel(UnconditionalDiffusionModel) :
    def __init__(self, network : nn.Module, sampler : nn.Module, auto_encoder : VariationalAutoEncoder, image_shape):
        super().__init__(network, sampler, image_shape)
        self.auto_encoder = auto_encoder
        self.auto_encoder.eval()
        for param in self.auto_encoder.parameters():
            param.requires_grad = False
        self.latent_shape = [*self.auto_encoder.decoder.z_shape[1:]]
        self.latent_shape[0] = self.auto_encoder.embed_dim
        
    def loss(self, x0):
        x0 = self.auto_encoder.encode(x0).sample()
        eps = torch.randn_like(x0)
        t = torch.randint(0, self.T, (x0.size(0),), device = x0.device)
        x_t = self.sampler.q_sample(x0, t, eps)
        eps_hat = self.network(x = x_t, t = t)
        return self.weighted_loss(t, eps, eps_hat)
            
    @torch.no_grad()
    def forward(self, n_samples : int = 4):
        x_T = torch.randn(n_samples, *self.latent_shape, device = next(self.buffers(), None).device )
        sample = self.sampler(x_T = x_T)
        return self.auto_encoder.decode(sample)
    
    @torch.no_grad()
    def generate_sequence(self, n_samples : int = 4):
        x_T = torch.randn(n_samples, *self.latent_shape, device = next(self.buffers(), None).device )
        sample_sequence = self.sampler.reverse_process(x_T, only_last=False)
        return sample_sequence