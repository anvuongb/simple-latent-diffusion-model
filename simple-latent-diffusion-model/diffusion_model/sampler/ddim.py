#source: https://nn.labml.ai/diffusion/stable_diffusion/sampler/ddim.html
import torch
import torch.nn as nn
import numpy as np
import yaml
from tqdm import tqdm

from helper.util import extract
from helper.beta_generator import BetaGenerator

class DDIM(nn.Module):
    def __init__(self, config_path):
        super(DDIM, self).__init__()
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)['sampler']
        self.T = config['T']
        beta_generator = BetaGenerator(T=self.T)
        
        self.register_buffer('timesteps', torch.linspace(0, 999, steps = 1000, dtype = torch.int))
        self.register_buffer('betas', getattr(beta_generator,
                                              f"{config['beta']}_beta_schedule",
                                              beta_generator.linear_beta_schedule)())
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alpha_bars', torch.cumprod(self.alphas, dim = 0))

        step = self.T // config['ddim_T']
        timesteps = torch.tensor(np.asarray(list(range(0, self.T, step))) + 1)
        self.ddim_timesteps = timesteps
        self.ddim_alpha = self.alpha_bars[self.ddim_timesteps].clone()
        self.ddim_alpha_sqrt = torch.sqrt(self.ddim_alpha)
        self.ddim_alpha_prev = torch.cat([self.alpha_bars[0:1], self.alpha_bars[self.ddim_timesteps[:-1]]])
        self.ddim_sigma = (config['ddim_eta'] * 
                            torch.sqrt((1-self.ddim_alpha_prev) / (1-self.ddim_alpha) *
                            (1 - self.ddim_alpha / self.ddim_alpha_prev)))
        self.ddim_sqrt_one_minus_alpha = torch.sqrt(1. - self.ddim_alpha)
        
    def set_network(self, network : nn.Module):
        self.network = network
        
    def q_sample(self, x0, t, eps = None):
        alpha_t_bar = extract(self.alpha_bars, t, x0.shape)
        if eps is None:
            eps = torch.randn_like(x0)
        noisy = alpha_t_bar.sqrt() * x0 + (1 - alpha_t_bar).sqrt() * eps
        return noisy
    
    @torch.no_grad()
    def get_x_prev_and_pred_x0(self, eps_t, index, x, temperature) :
        alpha = self.ddim_alpha[index]
        alpha_prev = self.ddim_alpha_prev[index]
        sigma = self.ddim_sigma[index]
        sqrt_one_minus_alpha = self.ddim_sqrt_one_minus_alpha[index]
        pred_x0 = (x - sqrt_one_minus_alpha * eps_t) / (alpha ** 0.5)
        dir_xt = (1. - alpha_prev - sigma ** 2).sqrt() * eps_t
        if sigma == 0. : noise = 0.
        else : noise = torch.randn_like(x, device = x.device)
        noise = noise * temperature
        x_prev = (alpha_prev ** 0.5) * pred_x0 + dir_xt + sigma * noise
        return x_prev

    @torch.no_grad()
    def reverse_process(self, x_T, temperature, **kwargs):
        x = x_T
        for i, t in tqdm(enumerate(reversed(self.ddim_timesteps))):
            index = len(self.ddim_timesteps) - i - 1
            x = self.p_sample(x, t, index, temperature, **kwargs)
        return x
    
    @torch.no_grad()
    def forward(self, x_T, temperature = 1., **kwargs):
        return self.reverse_process(x_T, temperature, **kwargs)
    
    @torch.no_grad()
    def p_sample(self, x, t, index, temperature, **kwargs):
        eps_hat = self.network(x = x, t = t, **kwargs)
        x_prev = self.get_x_prev_and_pred_x0(eps_hat, index, x, temperature = temperature)
        return x_prev