import torch
import torch.nn as nn

from diffusion_model.models.cond_diffusion_model import ConditionalDiffusionModel
from auto_encoder.models.variational_auto_encoder import VariationalAutoEncoder

class LatentDiffusionModel(ConditionalDiffusionModel) :
    def __init__(self, network : nn.Module, sampler : nn.Module, auto_encoder : VariationalAutoEncoder, image_shape):
        super().__init__(network, sampler, image_shape)
        self.auto_encoder = auto_encoder
        self.latent_shape = [*self.auto_encoder.decoder.z_shape[1:]]
        self.latent_shape[0] = self.auto_encoder.embed_dim
        
    def loss(self, x0, cond):
        x0 = self.auto_encoder.encode(x0).sample()
        eps = torch.randn_like(x0)
        t = torch.randint(0, self.T, (x0.size(0),), device = x0.device)
        x_t = self.sampler.q_sample(x0, t, eps)
        eps_hat = self.network(x = x_t, t = t, cond = cond)
        loss = nn.functional.mse_loss(eps, eps_hat)        
        return loss
            
    @torch.no_grad()
    def forward(self, cond, n_samples : int = 4):
        x_T = torch.randn(n_samples, *self.latent_shape, device = next(self.buffers(), None).device )
        cond = torch.tensor(cond, device = x_T.device)
        cond = cond.repeat(n_samples, 1)
        sample = self.sampler(self.network, x_T = x_T, cond = cond)
        print(sample)
        
        return self.auto_encoder.decode(sample)