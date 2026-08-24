import torch
import torch.nn as nn
from tqdm import tqdm
from config import Config

def extract(a, t):
    batch_size = t.shape[0]
    out = a.gather(-1, t) 
    return out.view(batch_size, 1, 1, 1)

# ========================================== #
# DDPM
class DDPM(nn.Module):
    def __init__(self, model, timesteps=1000):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        
        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('posterior_variance', betas)

    def q_sample(self, x_start, t, noise):
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t)
        x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t

    @torch.no_grad()
    def p_sample(self, x, t, cond, cfg_scale=3.0):
        noise_cond = self.model(x, t, cond)
        uncond = torch.zeros_like(cond)
        noise_uncond = self.model(x, t, uncond)
        predicted_noise = noise_uncond + cfg_scale * (noise_cond - noise_uncond)

        betas_t = extract(self.betas, t)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t)
        model_mean = sqrt_recip_alphas_t * (x - betas_t / sqrt_one_minus_alphas_cumprod_t * predicted_noise)

        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t)
            z = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * z

    @torch.no_grad()
    def sample(self, cond, cfg_scale=3.0, img_size=(3, 64, 64)):
        device = next(self.model.parameters()).device
        b = cond.shape[0]
        img = torch.randn((b, *img_size), device=device)
        
        for i in tqdm(reversed(range(self.timesteps)), desc="DDPM Sampling", total=self.timesteps, leave=False):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, cond, cfg_scale=cfg_scale)
        return img

# ========================================== #
# DDIM
class DDIM(DDPM):
    def __init__(self, model, timesteps=1000, ddim_timesteps=50, eta=0.0):
        super().__init__(model, timesteps)
        self.ddim_timesteps = ddim_timesteps
        self.eta = eta

    @torch.no_grad()
    def sample(self, cond, cfg_scale=5.0, img_size=(3, 64, 64)):
        device = next(self.model.parameters()).device
        b = cond.shape[0]
        img = torch.randn((b, *img_size), device=device)

        step_size = self.timesteps // self.ddim_timesteps
        time_steps = torch.arange(0, self.timesteps, step_size, device=device).long()
        time_steps = torch.flip(time_steps, dims=[0]) 

        for i, step in enumerate(tqdm(time_steps, desc=f"DDIM Sampling ({self.ddim_timesteps} steps)", leave=False)):
            t = torch.full((b,), step, device=device, dtype=torch.long)
            prev_step = step - step_size

            noise_cond = self.model(img, t, cond)
            uncond = torch.zeros_like(cond)
            noise_uncond = self.model(img, t, uncond)
            predicted_noise = noise_uncond + cfg_scale * (noise_cond - noise_uncond)

            alpha_bar_t = extract(self.alphas_cumprod, t)
            if prev_step >= 0:
                t_prev = torch.full((b,), prev_step, device=device, dtype=torch.long)
                alpha_bar_t_prev = extract(self.alphas_cumprod, t_prev)
            else:
                alpha_bar_t_prev = torch.ones_like(alpha_bar_t)

            pred_x0 = (img - torch.sqrt(1. - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
            sigma_t = self.eta * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev))
            dir_xt = torch.sqrt(1. - alpha_bar_t_prev - sigma_t**2) * predicted_noise
            
            noise = torch.randn_like(img) if step > 0 else torch.zeros_like(img)
            img = torch.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt + sigma_t * noise
        return img