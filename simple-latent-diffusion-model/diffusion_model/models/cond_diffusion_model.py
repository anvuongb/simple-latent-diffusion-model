import torch
import torch.nn as nn

from diffusion_model.models.uncond_diffusion_model import UnconditionalDiffusionModel

class ConditionalDiffusionModel(UnconditionalDiffusionModel) :
    def __init__(self, network : nn.Module, sampler : nn.Module, image_shape):
        super().__init__(network, sampler, image_shape)
        
    def loss(self, x0, cond, loss_weight = 1.0):
        eps = torch.randn_like(x0)
        t = torch.randint(0, self.T, (x0.size(0),), device = x0.device)
        x_t = self.sampler.q_sample(x0, t, eps)
        eps_hat = self.network(x = x_t, t = t, cond = cond)
        return self.weighted_loss(eps, eps_hat, loss_weight)
            
    @torch.no_grad()
    def forward(self, cond, n_samples : int = 4):
        x_T = torch.randn(n_samples, *self.image_shape, device = next(self.buffers(), None).device )
        cond = torch.tensor(cond, device = x_T.device)
        cond = cond.repeat(n_samples, 1)
        return self.sampler(x_T = x_T, cond = cond)