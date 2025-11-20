from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from delta_e2e.gen.model_diffusion import ConditionalGraphDenoiser, GaussianDiffusion


@dataclass
class DiffusionBundle:
    model: ConditionalGraphDenoiser
    diffusion: GaussianDiffusion
    feature_dim: int
    edge_channels: int
    code_feature_dim: int
    timesteps: int
    node_loss_weight: float
    edge_loss_weight: float


@dataclass
class GeneratedGraph:
    node_features: torch.Tensor
    adj_logits: Optional[torch.Tensor]
    adj_probs: Optional[torch.Tensor]
    adj_binary: Optional[torch.Tensor]
    edge_attr: Optional[torch.Tensor]
    edge_mask: torch.Tensor

    def edge_index(self) -> torch.Tensor:
        if self.adj_binary is None:
            raise ValueError("No edge samples available; adjacency was not generated.")
        return self.adj_binary.nonzero(as_tuple=False)

    def edge_attr_list(self) -> torch.Tensor:
        if self.edge_attr is None or self.adj_binary is None:
            raise ValueError("No edge samples available; edge attributes were not generated.")
        return self.edge_attr[self.adj_binary.bool()]


def load_diffusion_bundle(ckpt_path: str, device: torch.device) -> DiffusionBundle:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    feature_dim = int(ckpt["feature_dim"])
    hidden_dim = int(ckpt["hidden_dim"])
    time_embed_dim = int(ckpt["time_embed_dim"])
    timesteps = int(ckpt["timesteps"])
    code_feature_dim = int(ckpt.get("code_feature_dim", 16))
    edge_channels = int(ckpt.get("edge_channels", 1))

    model = ConditionalGraphDenoiser(feature_dim=feature_dim, edge_channels=edge_channels,
                                     hidden_dim=hidden_dim, time_embed_dim=time_embed_dim,
                                     code_feature_dim=code_feature_dim).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    diffusion = GaussianDiffusion(timesteps=timesteps)

    return DiffusionBundle(
        model=model,
        diffusion=diffusion,
        feature_dim=feature_dim,
        edge_channels=edge_channels,
        code_feature_dim=code_feature_dim,
        timesteps=timesteps,
        node_loss_weight=float(ckpt.get("node_loss_weight", 1.0)),
        edge_loss_weight=float(ckpt.get("edge_loss_weight", 1.0)),
    )


@torch.no_grad()
def sample_conditioned_graph(
    bundle: DiffusionBundle,
    sample: Dict,
    device: torch.device,
    edge_threshold: float = 0.5,
    nodes_only: bool = False,
) -> GeneratedGraph:
    model = bundle.model
    diffusion = bundle.diffusion

    x_cond = sample["x_cond"].unsqueeze(0).to(device)
    code_cond = sample["code_cond"].unsqueeze(0).to(device)
    node_mask = sample["mask"].unsqueeze(0).to(device)
    edge_mask = sample["edge_mask"].unsqueeze(0).to(device)

    edge_mask_cpu = edge_mask.squeeze(0).cpu()

    if nodes_only:
        nodes_gen = diffusion.p_sample_loop(
            model.node_denoiser,
            x_cond=x_cond,
            code_cond=code_cond,
            mask=node_mask,
            device=device,
        )
        nodes_gen = nodes_gen.squeeze(0).cpu()
        return GeneratedGraph(
            node_features=nodes_gen,
            adj_logits=None,
            adj_probs=None,
            adj_binary=None,
            edge_attr=None,
            edge_mask=edge_mask_cpu,
        )

    nodes_gen, edges_gen = diffusion.p_sample_graph(
        model,
        x_cond=x_cond,
        code_cond=code_cond,
        node_mask=node_mask,
        edge_mask=edge_mask,
        device=device,
    )

    nodes_gen = nodes_gen.squeeze(0).cpu()
    edges_gen = edges_gen.squeeze(0).cpu()

    adj_logits = edges_gen[..., 0]
    adj_probs = torch.sigmoid(adj_logits) * edge_mask_cpu.float()
    adj_binary = (adj_probs >= edge_threshold).int() * edge_mask_cpu.int()
    edge_attr = edges_gen[..., 1:]

    return GeneratedGraph(
        node_features=nodes_gen,
        adj_logits=adj_logits,
        adj_probs=adj_probs,
        adj_binary=adj_binary,
        edge_attr=edge_attr,
        edge_mask=edge_mask_cpu,
    )
