import torch
import torch.nn as nn
from tqdm import tqdm
import yaml

from helper.util import extract
from helper.beta_generator import BetaGenerator

class DDPM(nn.Module):
    def __init__(self, config_path):
        super(DDPM, self).__init__()
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
        
    def set_network(self, network : nn.Module):
        self.network = network
        
    def q_sample(self, x0, t, eps = None):
        alpha_t_bar = extract(self.alpha_bars, t, x0.shape)
        if eps is None:
            eps = torch.randn_like(x0)
        noisy = alpha_t_bar.sqrt() * x0 + (1 - alpha_t_bar).sqrt() * eps
        return noisy
    
    def sigma_t(self, t, alpha_t_bar): 
        beta_t = self.betas[t]
        prev_alpha_t_bar = self.alpha_bars[t - 1] if t > 1 else 1
        beta_tilda_t = ((1 - prev_alpha_t_bar)/(1 - alpha_t_bar)) * beta_t
        sigma_t = beta_tilda_t.sqrt()
        return sigma_t

    @torch.no_grad()
    def reverse_process(self, x_T, only_last=False, **kwargs):
        x = x_T
        if only_last:
            for t in tqdm(reversed(self.timesteps)):
                x = self.p_sample(x, t, **kwargs)
            return x
        else:
            x_seq = []
            x_seq.append(x)
            for t in tqdm(reversed(self.timesteps)):
                x_seq.append(self.p_sample(x_seq[-1], t, **kwargs))
            return x_seq    
    
    @torch.no_grad()
    def forward(self, x_T, **kwargs):
        return self.reverse_process(x_T, **kwargs)
    
    @torch.no_grad()
    def p_sample(self, x, t, **kwargs):
        eps_hat = self.network(x = x, t = t, **kwargs)
        alpha_t = self.alphas[t]
        alpha_t_bar = self.alpha_bars[t]
        x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eps_hat)
        z = torch.randn_like(x) if t > 0 else 0.
        x = x + self.sigma_t(t, alpha_t_bar) * z
        return x
    
    