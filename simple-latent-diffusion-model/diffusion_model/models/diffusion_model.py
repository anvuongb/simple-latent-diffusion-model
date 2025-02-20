import torch
import torch.nn as nn

class DiffusionModel(nn.Module) :
    def __init__(self, network : nn.Module, sampler : nn.Module, image_shape):
        super().__init__()
        self.add_module('sampler', sampler)
        self.add_module('network', network)
        self.sampler.set_network(network)
        self.T = sampler.T
        self.image_shape = image_shape
        
    def loss(self, x0, **kwargs):
        eps = torch.randn_like(x0)
        t = torch.randint(0, self.T, (x0.size(0),), device = x0.device)
        x_t = self.sampler.q_sample(x0, t, eps)
        eps_hat = self.network(x = x_t, t = t, **kwargs)
        return nn.functional.mse_loss(eps, eps_hat)
            
    @torch.no_grad()
    def forward(self, n_samples : int = 4, **kwargs):
        x_T = torch.randn(n_samples, *self.image_shape, device = next(self.buffers(), None).device)
        return self.sampler(x_T = x_T, **kwargs)