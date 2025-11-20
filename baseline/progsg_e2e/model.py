"""ProgSG baseline architectures adapted to the ForgeHLS e2e dataset."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import (
    GlobalAttention,
    JumpingKnowledge,
    TransformerConv,
    global_add_pool,
)
from torch_geometric.nn.norm import LayerNorm

# ---------------------------------------------------------------------------
# Import feature-dimension helpers from delta_e2e utilities
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DELTA_UTILS_DIR = REPO_ROOT / "delta_e2e"
if str(DELTA_UTILS_DIR) not in sys.path:
    sys.path.append(str(DELTA_UTILS_DIR))

from utils import (  # type: ignore  # pylint: disable=wrong-import-position
    get_edge_feature_dims,
    get_node_feature_dims,
)


TARGET_NAMES: Tuple[str, ...] = ("dsp", "lut", "ff", "latency")
NODE_FEATURE_DIMS = get_node_feature_dims()
EDGE_FEATURE_DIMS = get_edge_feature_dims()

# Scalar attributes attached to each graph that provide pipeline/context cues.
SCALAR_ATTR_NAMES: Tuple[str, ...] = (
    "pragma_count",
    "has_pipeline",
    "pipeline_region_count",
    "avg_ii",
    "max_pipe_depth",
)


class CategoricalEmbedding(nn.Module):
    """Embed multi-field categorical features by summing separate embeddings."""

    def __init__(self, feature_dims: Iterable[int], embed_dim: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(num_embeddings=dim, embedding_dim=embed_dim)
            for dim in feature_dims
        )
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.long:
            x = x.long()
        pieces = [emb(x[:, idx]) for idx, emb in enumerate(self.embeddings)]
        stacked = torch.stack(pieces, dim=0)
        return stacked.sum(dim=0)


def _build_mlp(
    in_dim: int,
    hidden_dims: Sequence[int],
    out_dim: int,
    activation: str = "elu",
    dropout: float = 0.0,
) -> nn.Sequential:
    layers: List[nn.Module] = []
    last_dim = in_dim
    act: nn.Module
    for hidden in hidden_dims:
        if hidden <= 0:
            continue
        layers.append(nn.Linear(last_dim, hidden))
        if activation == "elu":
            act = nn.ELU()
        elif activation == "relu":
            act = nn.ReLU()
        elif activation == "gelu":
            act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        layers.append(act)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        last_dim = hidden
    layers.append(nn.Linear(last_dim, out_dim))
    return nn.Sequential(*layers)


class ProgSGStyleModel(nn.Module):
    """Graph encoder mirroring the best-performing ProgSG configuration."""

    def __init__(
        self,
        hidden_dim: int = 512,
        num_layers: int = 8,
        heads: int = 8,
        dropout: float = 0.1,
        activation: str = "elu",
        use_scalar_context: bool = True,
        with_readout: bool = True,
        targets: Sequence[str] = TARGET_NAMES,
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads for TransformerConv")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.activation = activation
        self.use_scalar_context = use_scalar_context
        self.with_readout = with_readout
        self.targets = list(targets)

        self.node_encoder = CategoricalEmbedding(NODE_FEATURE_DIMS, hidden_dim)
        self.edge_encoder = CategoricalEmbedding(EDGE_FEATURE_DIMS, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        out_channels = hidden_dim // heads
        for _ in range(num_layers):
            conv = TransformerConv(
                in_channels=hidden_dim,
                out_channels=out_channels,
                heads=heads,
                edge_dim=hidden_dim,
                dropout=dropout,
                beta=True,
            )
            self.convs.append(conv)
            self.norms.append(LayerNorm(hidden_dim))

        self.jk = JumpingKnowledge(mode="max")
        gate_hidden = max(hidden_dim // 2, 1)
        self.graph_gate = nn.Sequential(
            nn.Linear(hidden_dim, gate_hidden),
            nn.ELU(),
            nn.Linear(gate_hidden, 1),
        )
        self.graph_pool = GlobalAttention(gate_nn=self.graph_gate)

        if use_scalar_context:
            self.scalar_encoder = nn.Sequential(
                nn.Linear(len(SCALAR_ATTR_NAMES), hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.scalar_encoder = None

        if with_readout:
            self.target_heads = nn.ModuleDict(
                {
                    name: _build_mlp(
                        in_dim=hidden_dim,
                        hidden_dims=(hidden_dim // 2, hidden_dim // 4, hidden_dim // 8),
                        out_dim=1,
                        activation=activation,
                        dropout=dropout,
                    )
                    for name in self.targets
                }
            )
        else:
            self.target_heads = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(module.weight)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode_nodes(self, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.node_encoder(batch.x)
        edge_attr = None
        if batch.edge_attr is not None:
            edge_attr = self.edge_encoder(batch.edge_attr)
        h = x
        outputs: List[torch.Tensor] = []
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, batch.edge_index, edge_attr)
            h_new = F.elu(h_new) if self.activation == "elu" else getattr(F, self.activation)(h_new)
            h_new = norm(h_new)
            h = h + F.dropout(h_new, p=self.dropout, training=self.training)
            outputs.append(h)
        if len(outputs) == 1:
            final = outputs[0]
        else:
            final = self.jk(outputs)
        return final, edge_attr, batch.edge_index

    def _gather_scalar_context(self, batch: Batch, device: torch.device) -> Optional[torch.Tensor]:
        if not self.use_scalar_context or self.scalar_encoder is None:
            return None
        num_graphs = int(getattr(batch, "num_graphs", 0))
        if num_graphs == 0:
            return None
        features: List[torch.Tensor] = []
        for name in SCALAR_ATTR_NAMES:
            value = getattr(batch, name, None)
            if value is None:
                features.append(torch.zeros((num_graphs, 1), device=device))
                continue
            tensor = value.float().to(device)
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(1)
            features.append(tensor)
        return torch.cat(features, dim=1)

    def encode_graph(self, batch: Batch) -> torch.Tensor:
        node_states, _, _ = self.encode_nodes(batch)
        pooled = self.graph_pool(node_states, batch.batch)
        scalars = self._gather_scalar_context(batch, pooled.device)
        if scalars is not None:
            pooled = pooled + self.scalar_encoder(scalars)
        return pooled

    def forward(self, batch: Batch) -> torch.Tensor:
        graph_emb = self.encode_graph(batch)
        if not self.with_readout:
            return graph_emb
        if self.target_heads is None:
            raise RuntimeError("Readout requested but target_heads not initialised")
        preds = [
            self.target_heads[target](graph_emb) for target in self.targets
        ]
        return torch.cat(preds, dim=1)


class ProgSGMultimodalModel(nn.Module):
    """ProgSG variant that fuses graph and source-code representations."""

    def __init__(
        self,
        hidden_dim: int = 512,
        num_layers: int = 8,
        heads: int = 8,
        dropout: float = 0.1,
        activation: str = "elu",
        code_embedding_dim: int = 768,
        code_transformer_layers: int = 2,
        code_transformer_heads: int = 8,
        fusion_mode: str = "concat",
        node_token_interaction: bool = False,
        targets: Sequence[str] = TARGET_NAMES,
    ) -> None:
        super().__init__()
        if fusion_mode not in {"concat", "add"}:
            raise ValueError(f"Unsupported fusion mode: {fusion_mode}")

        self.graph_encoder = ProgSGStyleModel(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            activation=activation,
            use_scalar_context=True,
            with_readout=False,
            targets=targets,
        )

        self.code_projection = nn.Linear(code_embedding_dim, hidden_dim)

        if code_transformer_layers > 0:
            transformer_activation = activation if activation in {"relu", "gelu"} else "relu"
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=code_transformer_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation=transformer_activation,
            )
            self.code_transformer: Optional[nn.TransformerEncoder] = nn.TransformerEncoder(
                encoder_layer,
                num_layers=code_transformer_layers,
            )
        else:
            self.code_transformer = None

        self.fusion_mode = fusion_mode
        self.node_token_interaction = node_token_interaction
        fusion_input_dim = hidden_dim * 2 if fusion_mode == "concat" else hidden_dim

        self.fusion_head = nn.ModuleDict(
            {
                name: _build_mlp(
                    in_dim=fusion_input_dim,
                    hidden_dims=(hidden_dim, hidden_dim // 2, hidden_dim // 4),
                    out_dim=1,
                    activation=activation,
                    dropout=dropout,
                )
                for name in targets
            }
        )
        self.targets = list(targets)

    def forward(self, batch: Batch) -> torch.Tensor:
        graph_emb = self.graph_encoder.encode_graph(batch)

        if self.node_token_interaction:
            node_states, _, _ = self.graph_encoder.encode_nodes(batch)
            tokens_info = self._get_projected_tokens(batch, device=node_states.device)
            if tokens_info is not None:
                token_proj, seq_lengths, starts, offsets = tokens_info
                fused_nodes = node_states.clone()
                for gidx in range(int(seq_lengths.size(0))):
                    node_mask = (batch.batch == gidx)
                    if not torch.any(node_mask):
                        continue
                    start = int(starts[gidx].item())
                    end = int(offsets[gidx].item())
                    if end <= start:
                        continue
                    nodes_h = torch.nan_to_num(node_states[node_mask], nan=0.0, posinf=0.0, neginf=0.0)
                    tokens_slice = torch.nan_to_num(token_proj[start:end], nan=0.0, posinf=0.0, neginf=0.0)
                    scale = 1.0 / math.sqrt(max(nodes_h.size(-1), 1))
                    attn_logits = torch.matmul(nodes_h, tokens_slice.transpose(0, 1)) * scale
                    attn_logits = torch.nan_to_num(attn_logits, nan=0.0, posinf=1e4, neginf=-1e4)
                    attn_logits = attn_logits - attn_logits.max(dim=1, keepdim=True).values
                    attn = F.softmax(attn_logits, dim=1)
                    attn = torch.nan_to_num(attn, nan=0.0)
                    node_enh = torch.matmul(attn, tokens_slice)
                    fused_nodes[node_mask] = nodes_h + node_enh
                graph_emb = self.graph_encoder.graph_pool(fused_nodes, batch.batch)

        code_emb = self._encode_code(batch, device=graph_emb.device)

        if code_emb is None:
            if self.fusion_mode == "concat":
                fused = torch.cat([graph_emb, torch.zeros_like(graph_emb)], dim=-1)
            else:
                fused = graph_emb
        elif self.fusion_mode == "concat":
            fused = torch.cat([graph_emb, code_emb], dim=-1)
        else:
            fused = graph_emb + code_emb

        preds = [self.fusion_head[target](fused) for target in self.targets]
        return torch.cat(preds, dim=1)

    def _encode_code(self, batch: Batch, device: torch.device) -> Optional[torch.Tensor]:
        if not hasattr(batch, "code_seq_len"):
            return None

        seq_lengths = batch.code_seq_len.view(-1)
        num_graphs = int(seq_lengths.size(0))
        total_tokens = int(seq_lengths.sum().item())
        if total_tokens == 0 or num_graphs == 0:
            return None

        pooled = getattr(batch, "code_pooled_embedding", None)
        if pooled is not None:
            pooled = pooled.to(device)
            if pooled.dim() == 1:
                pooled = pooled.unsqueeze(0)
            if pooled.size(0) != num_graphs:
                raise ValueError("Mismatch between number of graphs and pooled code embeddings.")
            projected = self.code_projection(pooled)
            return projected

        token_embeddings = getattr(batch, "code_token_embeddings", None)
        attention_mask = getattr(batch, "code_attention_mask", None)
        if token_embeddings is None or attention_mask is None:
            return None

        token_embeddings = token_embeddings.to(device)
        attention_mask = attention_mask.to(device)

        offsets = torch.cumsum(seq_lengths, dim=0)
        starts = offsets - seq_lengths
        embeddings: List[torch.Tensor] = []

        for graph_idx in range(num_graphs):
            start = int(starts[graph_idx].item())
            end = int(offsets[graph_idx].item())
            if start == end:
                embeddings.append(torch.zeros(self.code_projection.out_features, device=device))
                continue
            tokens_slice = token_embeddings[start:end]
            attn_slice = attention_mask[start:end]
            projected = self.code_projection(tokens_slice)
            if self.code_transformer is not None and projected.size(0) > 0:
                key_padding_mask = (attn_slice == 0).unsqueeze(0)
                projected = self.code_transformer(
                    projected.unsqueeze(0),
                    src_key_padding_mask=key_padding_mask,
                ).squeeze(0)
            mask = attn_slice.unsqueeze(-1).to(projected.dtype)
            denom = mask.sum(dim=0).clamp_min(1.0)
            pooled_vec = (projected * mask).sum(dim=0) / denom
            embeddings.append(pooled_vec)

        return torch.stack(embeddings, dim=0)

    def _get_projected_tokens(
        self,
        batch: Batch,
        device: torch.device,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        if not hasattr(batch, "code_seq_len"):
            return None
        seq_lengths = batch.code_seq_len.view(-1)
        num_graphs = int(seq_lengths.size(0))
        total_tokens = int(seq_lengths.sum().item())
        if total_tokens == 0 or num_graphs == 0:
            return None

        token_embeddings = getattr(batch, "code_token_embeddings", None)
        attention_mask = getattr(batch, "code_attention_mask", None)
        if token_embeddings is None or attention_mask is None:
            return None

        token_embeddings = token_embeddings.to(device)
        attention_mask = attention_mask.to(device)

        offsets = torch.cumsum(seq_lengths, dim=0)
        starts = offsets - seq_lengths
        projected_list: List[torch.Tensor] = []

        for graph_idx in range(num_graphs):
            start = int(starts[graph_idx].item())
            end = int(offsets[graph_idx].item())
            if start == end:
                continue
            tokens_slice = token_embeddings[start:end]
            attn_slice = attention_mask[start:end]
            projected = self.code_projection(tokens_slice)
            if self.code_transformer is not None and projected.size(0) > 0:
                key_padding_mask = (attn_slice == 0).unsqueeze(0)
                projected = self.code_transformer(
                    projected.unsqueeze(0),
                    src_key_padding_mask=key_padding_mask,
                ).squeeze(0)
            projected_list.append(projected)

        if not projected_list:
            return None

        token_proj = torch.cat(projected_list, dim=0)
        return token_proj, seq_lengths.to(device), starts.to(device), offsets.to(device)


__all__ = [
    "ProgSGStyleModel",
    "ProgSGMultimodalModel",
    "TARGET_NAMES",
]
