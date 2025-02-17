import torch

from diffusion_model.sampler.base_sampler import BaseSampler

class DDPM(BaseSampler):
    def __init__(self, config_path):
        super().__init__(config_path)
        self.sigma = (((1 - self.alpha_bar_prev) / (1 - self.alpha_bar)) * self.beta).sqrt()

    @torch.no_grad()
    def get_x_prev(self, x, t, eps_hat):
        x = (1 / self.alpha_sqrt[t]) \
           * (x - (self.beta[t] / self.sqrt_one_minus_alpha_bar[t] * eps_hat))
        z = torch.randn_like(x) if t > 0 else 0.
        x = x + self.sigma[t] * z
        return x
    