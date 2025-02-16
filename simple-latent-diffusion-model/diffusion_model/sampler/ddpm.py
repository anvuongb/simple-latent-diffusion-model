import torch
from diffusion_model.sampler.base_sampler import BaseSampler

class DDPM(BaseSampler):
    def __init__(self, config_path):
        super(config_path).__init__()
        
    def sigma_t(self, t): 
        alpha_t_bar = self.alpha_bars[t]
        beta_t = self.betas[t]
        prev_alpha_t_bar = self.alpha_bars[t - 1] if t > 1 else 1
        beta_tilda_t = ((1 - prev_alpha_t_bar)/(1 - alpha_t_bar)) * beta_t
        sigma_t = beta_tilda_t.sqrt()
        return sigma_t

    @torch.no_grad()
    def get_x_prev(self, x, t, index, eps_hat):
        # index == t!
        alpha_t = self.alphas[t]
        alpha_t_bar = self.alpha_bars[t]
        x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eps_hat)
        z = torch.randn_like(x) if t > 0 else 0.
        x = x + self.sigma_t(t) * z
        return x
    