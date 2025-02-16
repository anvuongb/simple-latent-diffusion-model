#source: https://nn.labml.ai/diffusion/stable_diffusion/sampler/ddim.html
import torch
import numpy as np

from diffusion_model.sampler.base_sampler import BaseSampler

class DDIM(BaseSampler):
    def __init__(self, config_path):
        super(config_path).__init__()

        step = self.T // self.config['ddim_T']
        self.timesteps = torch.tensor(np.asarray(list(range(0, self.T, step))) + 1)
        self.ddim_alpha = self.alpha_bars[self.timesteps].clone()
        self.ddim_alpha_sqrt = torch.sqrt(self.ddim_alpha)
        self.ddim_alpha_prev = torch.cat([self.alpha_bars[0:1], self.alpha_bars[self.timesteps[:-1]]])
        self.ddim_sigma = (self.config['ddim_eta'] * 
                            torch.sqrt((1-self.ddim_alpha_prev) / (1-self.ddim_alpha) *
                            (1 - self.ddim_alpha / self.ddim_alpha_prev)))
        self.ddim_sqrt_one_minus_alpha = torch.sqrt(1. - self.ddim_alpha)
    
    def sigma_t(self, t):
        return self.ddim_sigma[t]

    def get_x_prev(self, x, t, index, eps_hat) :
        alpha = self.ddim_alpha[index]
        alpha_prev = self.ddim_alpha_prev[index]
        sigma = self.sigma_t(index)
        sqrt_one_minus_alpha = self.ddim_sqrt_one_minus_alpha[index]

        pred_x0 = (x - sqrt_one_minus_alpha * eps_hat) / (alpha ** 0.5)
        dir_xt = (1. - alpha_prev - sigma ** 2).sqrt() * eps_hat
        if sigma == 0. : noise = 0.
        else : noise = torch.randn_like(x, device = x.device)
        x = (alpha_prev ** 0.5) * pred_x0 + dir_xt + sigma * noise
        return x