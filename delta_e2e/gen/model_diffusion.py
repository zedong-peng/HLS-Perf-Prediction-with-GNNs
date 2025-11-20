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
    """Per-node conditional denoiser operating on padded tensors [B, N, F]."""

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

    def forward(
        self,
        x_t: torch.Tensor,
        x_cond: torch.Tensor,
        code_cond: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = x_t.shape
        time_emb = sinusoidal_time_embedding(t, self.time_embed_dim).unsqueeze(1).expand(B, N, self.time_embed_dim)
        h = torch.cat([x_t, x_cond, code_cond, time_emb], dim=-1)
        eps_pred = self.net(h)
        if mask is not None:
            eps_pred = eps_pred * mask.unsqueeze(-1).float()
        return eps_pred


class ConditionalEdgeDenoiser(nn.Module):
    """Edge-level denoiser on dense adjacency tensors [B, N, N, C]."""

    def __init__(
        self,
        feature_dim: int,
        edge_channels: int,
        hidden_dim: int = 256,
        time_embed_dim: int = 128,
        code_feature_dim: int = 16,
    ) -> None:
        super().__init__()
        self.edge_channels = int(edge_channels)
        self.time_embed_dim = int(time_embed_dim)
        self.feature_dim = int(feature_dim)
        self.code_feature_dim = int(code_feature_dim)

        pair_cond_dim = 2 * (feature_dim + code_feature_dim)
        input_dim = edge_channels + pair_cond_dim + time_embed_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, edge_channels),
        )

    def forward(
        self,
        edge_t: torch.Tensor,
        x_cond: torch.Tensor,
        code_cond: torch.Tensor,
        t: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
        edge_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _, _ = edge_t.shape
        time_emb = sinusoidal_time_embedding(t, self.time_embed_dim).view(B, 1, 1, self.time_embed_dim).expand(B, N, N, self.time_embed_dim)

        node_context = torch.cat([x_cond, code_cond], dim=-1)
        src = node_context.unsqueeze(2).expand(B, N, N, -1)
        dst = node_context.unsqueeze(1).expand(B, N, N, -1)
        pair_cond = torch.cat([src, dst], dim=-1)

        h = torch.cat([edge_t, pair_cond, time_emb], dim=-1)
        eps_pred = self.net(h.view(B, N * N, -1)).view(B, N, N, self.edge_channels)

        mask = edge_mask
        if mask is None and node_mask is not None:
            mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        if mask is not None:
            eps_pred = eps_pred * mask.unsqueeze(-1).float()
        return eps_pred


class ConditionalGraphDenoiser(nn.Module):
    """Joint node + edge conditional denoiser."""

    def __init__(
        self,
        feature_dim: int,
        edge_channels: int,
        hidden_dim: int = 256,
        time_embed_dim: int = 128,
        code_feature_dim: int = 16,
    ) -> None:
        super().__init__()
        self.node_denoiser = ConditionalNodeDenoiser(feature_dim, hidden_dim, time_embed_dim, code_feature_dim)
        self.edge_denoiser = ConditionalEdgeDenoiser(feature_dim, edge_channels, hidden_dim, time_embed_dim, code_feature_dim)
        self.edge_channels = int(edge_channels)

    def forward(
        self,
        x_t: torch.Tensor,
        edge_t: torch.Tensor,
        x_cond: torch.Tensor,
        code_cond: torch.Tensor,
        t: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
        edge_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        node_eps = self.node_denoiser(x_t, x_cond, code_cond, t, node_mask)
        edge_eps = self.edge_denoiser(edge_t, x_cond, code_cond, t, node_mask, edge_mask)
        return {"node": node_eps, "edge": edge_eps}


class GaussianDiffusion:
    """DDPM utilities for continuous node/edge tensors."""

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
        shape = [x0.size(0)] + [1] * (x0.dim() - 1)
        a_bar = alphas_cumprod[t].view(*shape)
        xt = torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise
        return xt, noise

    def _prepare_mask(self, mask: Optional[torch.Tensor], target_shape: torch.Size) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        mask_f = mask.float()
        while mask_f.dim() < len(target_shape):
            mask_f = mask_f.unsqueeze(-1)
        return mask_f

    def _denoise_step(self, x: torch.Tensor, eps: torch.Tensor, step: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = x.device
        beta_t = self.betas[step].to(device)
        alpha_t = self.alphas[step].to(device)
        alpha_bar_t = self.alphas_cumprod[step].to(device)

        mean = (x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps) / torch.sqrt(alpha_t)
        mask_f = self._prepare_mask(mask, x.shape)

        if step > 0:
            noise = torch.randn_like(x)
            if mask_f is not None:
                noise = noise * mask_f
            x_next = mean + torch.sqrt(beta_t) * noise
        else:
            x_next = mean

        if mask_f is not None:
            x_next = x_next * mask_f
        return x_next

    @torch.no_grad()
    def p_sample_loop(self, model: nn.Module, x_cond: torch.Tensor, code_cond: torch.Tensor, mask: torch.Tensor, device: torch.device) -> torch.Tensor:
        B, N, Fdim = x_cond.shape
        x = torch.randn(B, N, Fdim, device=device)
        for step in reversed(range(self.timesteps)):
            t = torch.full((B,), step, device=device, dtype=torch.long)
            eps = model(x, x_cond, code_cond, t, mask)
            x = self._denoise_step(x, eps, step, mask)
        return x

    @torch.no_grad()
    def p_sample_graph(
        self,
        model: ConditionalGraphDenoiser,
        x_cond: torch.Tensor,
        code_cond: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, Fdim = x_cond.shape
        edge_channels = model.edge_channels
        node_state = torch.randn(B, N, Fdim, device=device)
        edge_state = torch.randn(B, N, N, edge_channels, device=device)

        for step in reversed(range(self.timesteps)):
            t = torch.full((B,), step, device=device, dtype=torch.long)
            outputs = model(node_state, edge_state, x_cond, code_cond, t, node_mask, edge_mask)
            node_state = self._denoise_step(node_state, outputs["node"], step, node_mask)
            edge_state = self._denoise_step(edge_state, outputs["edge"], step, edge_mask)
        return node_state, edge_state


class NodeLatentEncoder(nn.Module):
    def __init__(self, feature_dim: int, code_feature_dim: int, latent_dim: int = 64, hidden_dim: int = 128) -> None:
        super().__init__()
        in_dim = int(feature_dim) + int(feature_dim) + int(code_feature_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self,
        x_target: torch.Tensor,
        x_cond: torch.Tensor,
        code_cond: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = torch.cat([x_target, x_cond, code_cond], dim=-1)
        z = self.net(h)
        if mask is not None:
            z = z * mask.unsqueeze(-1).float()
        return z


class NodeLatentDenoiser(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        feature_dim: int,
        code_feature_dim: int,
        hidden_dim: int = 256,
        time_embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.time_embed_dim = int(time_embed_dim)
        in_dim = int(latent_dim) + int(feature_dim) + int(code_feature_dim) + int(time_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self,
        z_t: torch.Tensor,
        x_cond: torch.Tensor,
        code_cond: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = z_t.shape
        time_emb = sinusoidal_time_embedding(t, self.time_embed_dim).unsqueeze(1).expand(B, N, self.time_embed_dim)
        h = torch.cat([z_t, x_cond, code_cond, time_emb], dim=-1)
        eps = self.net(h)
        if mask is not None:
            eps = eps * mask.unsqueeze(-1).float()
        return eps
