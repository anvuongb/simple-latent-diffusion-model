import torch
import torch.nn as nn
from einops import reduce
from helper.util import extract

class UnconditionalDiffusionModel(nn.Module) :
    def __init__(self, network : nn.Module, sampler : nn.Module, image_shape):
        super().__init__()
        self.add_module('sampler', sampler)
        self.add_module('network', network)
        self.sampler.set_network(network)
        self.T = sampler.T
        self.image_shape = image_shape
        alpha_bar = self.sampler.alpha_bar
        snr = alpha_bar / (1 - alpha_bar)
        maybe_clipped_snr = snr.clone()
        maybe_clipped_snr.clamp_(max = 5)
        self.register_buffer('loss_weight', maybe_clipped_snr / snr)
        
    def weighted_loss(self, t, eps, eps_hat):
        loss = nn.functional.mse_loss(eps, eps_hat, reduction='none')
        loss = reduce(loss, 'b ... -> b', 'mean')
        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()
        
    def loss(self, x0):
        eps = torch.randn_like(x0)
        t = torch.randint(0, self.T, (x0.size(0),), device = x0.device)
        x_t = self.sampler.q_sample(x0, t, eps)
        eps_hat = self.network(x = x_t, t = t)
        return self.weighted_loss(t, eps, eps_hat)
            
    @torch.no_grad()
    def forward(self, n_samples : int = 4):
        x_T = torch.randn(n_samples, *self.image_shape, device = next(self.buffers(), None).device )
        return self.sampler(x_T = x_T)