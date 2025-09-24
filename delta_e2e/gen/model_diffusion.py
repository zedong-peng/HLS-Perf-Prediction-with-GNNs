from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(
        -torch.arange(half, device=device, dtype=torch.float32)
        * (torch.log(torch.tensor(10000.0)) / max(1, half - 1))
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ConditionalNodeDenoiser(nn.Module):
    """Per-node conditional denoiser. Input is padded as [B, N, F].

    Accepts extra per-node code condition with dim C.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 256, time_embed_dim: int = 128, code_feature_dim: int = 16) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.time_embed_dim = int(time_embed_dim)
        self.code_feature_dim = int(code_feature_dim)

        input_dim = feature_dim * 2 + time_embed_dim + code_feature_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, x_t: torch.Tensor, x_cond: torch.Tensor, code_cond: torch.Tensor, t: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x_t, x_cond, code_cond: [B, N, *]; t: [B]
        B, N, _ = x_t.shape
        time_emb = sinusoidal_time_embedding(t, self.time_embed_dim).unsqueeze(1).expand(B, N, self.time_embed_dim)
        h = torch.cat([x_t, x_cond, code_cond, time_emb], dim=-1)
        eps_pred = self.net(h)
        if mask is not None:
            eps_pred = eps_pred * mask.unsqueeze(-1).float()
        return eps_pred


class GaussianDiffusion:
    """Simple DDPM utilities for continuous node features."""

    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2) -> None:
        self.timesteps = int(timesteps)
        betas = torch.linspace(beta_start, beta_end, self.timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register(betas=betas, alphas=alphas, alphas_cumprod=alphas_cumprod)

    def register(self, **tensors):
        for k, v in tensors.items():
            setattr(self, k, v)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        alphas_cumprod = self.alphas_cumprod.to(x0.device)
        a_bar = alphas_cumprod[t].view(-1, 1, 1)
        xt = (a_bar.sqrt() * x0) + ((1.0 - a_bar).sqrt() * noise)
        return xt, noise

    @torch.no_grad()
    def p_sample_loop(self, model: nn.Module, x_cond: torch.Tensor, code_cond: torch.Tensor, mask: torch.Tensor, device: torch.device) -> torch.Tensor:
        B, N, Fdim = x_cond.shape
        x = torch.randn(B, N, Fdim, device=device)
        for step in reversed(range(self.timesteps)):
            t = torch.full((B,), step, device=device, dtype=torch.long)
            eps = model(x, x_cond, code_cond, t, mask)
            beta_t = self.betas[step].to(device)
            alpha_t = 1.0 - beta_t
            alpha_bar_t = self.alphas_cumprod[step].to(device)
            coef = 1.0 / alpha_t.sqrt()
            mean = coef * (x - (beta_t / (1.0 - alpha_bar_t).sqrt()) * eps)
            if step > 0:
                noise = torch.randn_like(x)
                x = mean + beta_t.sqrt() * noise
            else:
                x = mean
            x = x * mask.unsqueeze(-1).float()
        return x 