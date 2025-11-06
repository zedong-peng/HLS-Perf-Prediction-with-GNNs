"""Simplified ProgSG-style graph neural network for QoR regression."""

from __future__ import annotations

from typing import Iterable, List, Optional

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import TransformerConv, global_add_pool
from torch_geometric.nn.norm import LayerNorm
import math

# ---------------------------------------------------------------------------
# Import feature dimension helpers from delta_e2e
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DELTA_UTILS_DIR = REPO_ROOT / "delta_e2e"
if str(DELTA_UTILS_DIR) not in sys.path:
    sys.path.append(str(DELTA_UTILS_DIR))

from utils import (  # type: ignore
    get_node_feature_dims,
    get_edge_feature_dims,
)

TARGET_NAMES = ["dsp", "lut", "ff", "latency"]
NODE_FEATURE_DIMS = get_node_feature_dims()
EDGE_FEATURE_DIMS = get_edge_feature_dims()

# Scalar attributes attached to each graph that can provide useful
# performance-context signals. These are expected to be present on the
# batched object with one value per graph (shape [num_graphs] or [num_graphs, 1]).
SCALAR_ATTR_NAMES: List[str] = [
    "pragma_count",
    "has_pipeline",
    "pipeline_region_count",
    "avg_ii",
    "max_pipe_depth",
]


class CategoricalEmbedding(nn.Module):
    """Embed multi-field categorical features by summing separate embeddings."""

    def __init__(self, feature_dims: Iterable[int], embed_dim: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_embeddings=dim, embedding_dim=embed_dim) for dim in feature_dims]
        )
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.long:
            x = x.long()
        pieces = [emb(x[:, idx]) for idx, emb in enumerate(self.embeddings)]
        stacked = torch.stack(pieces, dim=0)
        return stacked.sum(dim=0)


class ProgSGStyleModel(nn.Module):
    """Graph neural network with Transformer-style message passing."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        use_pragma_scalar: bool = True,
        with_readout: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads for TransformerConv")
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_pragma_scalar = use_pragma_scalar
        self.with_readout = with_readout

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

        if use_pragma_scalar:
            # Expand to inject a small vector of pipeline/context scalars.
            self.pragma_encoder = nn.Sequential(
                nn.Linear(len(SCALAR_ATTR_NAMES), hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.pragma_encoder = None

        if with_readout:
            self.readout = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, len(TARGET_NAMES)),
            )
        else:
            self.readout = None

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode_nodes(self, batch: Batch) -> torch.Tensor:
        x = self.node_encoder(batch.x)
        edge_attr = None
        if batch.edge_attr is not None:
            edge_attr = self.edge_encoder(batch.edge_attr)
        h = x
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, batch.edge_index, edge_attr)
            h_new = F.relu(h_new)
            h_new = norm(h_new)
            h = h + F.dropout(h_new, p=self.dropout, training=self.training)
        return h

    def _gather_pipeline_scalars(self, batch: Batch, device: torch.device) -> Optional[torch.Tensor]:
        if not self.use_pragma_scalar or self.pragma_encoder is None:
            return None
        num_graphs = int(getattr(batch, "num_graphs", 0))
        if num_graphs == 0:
            return None
        features: List[torch.Tensor] = []
        for name in SCALAR_ATTR_NAMES:
            if hasattr(batch, name):
                t = getattr(batch, name).float().to(device)
                if t.dim() == 1:
                    t = t.unsqueeze(1)
            else:
                t = torch.zeros((num_graphs, 1), device=device)
            features.append(t)
        return torch.cat(features, dim=1)

    def encode_graph(self, batch: Batch) -> torch.Tensor:
        h = self.encode_nodes(batch)
        pooled = global_add_pool(h, batch.batch)
        scalars = self._gather_pipeline_scalars(batch, pooled.device)
        if scalars is not None:
            pooled = pooled + self.pragma_encoder(scalars)
        return pooled

    def forward(self, batch: Batch) -> torch.Tensor:
        pooled = self.encode_graph(batch)
        if not self.with_readout:
            return pooled
        if self.readout is None:
            raise RuntimeError("Readout layer is disabled but forward was called expecting outputs")
        out = self.readout(pooled)
        return out


__all__ = [
    "ProgSGStyleModel",
    "ProgSGMultimodalModel",
    "TARGET_NAMES",
]


class ProgSGMultimodalModel(nn.Module):
    """ProgSG variant that fuses graph and source-code representations."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        code_embedding_dim: int = 768,
        code_transformer_layers: int = 2,
        code_transformer_heads: int = 4,
        fusion_mode: str = "concat",
        node_token_interaction: bool = False,
    ) -> None:
        super().__init__()
        if fusion_mode not in {"concat", "add"}:
            raise ValueError(f"Unsupported fusion mode: {fusion_mode}")

        self.graph_encoder = ProgSGStyleModel(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            use_pragma_scalar=True,
            with_readout=False,
        )

        self.code_projection = nn.Linear(code_embedding_dim, hidden_dim)

        if code_transformer_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=code_transformer_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.code_transformer: Optional[nn.TransformerEncoder] = nn.TransformerEncoder(
                encoder_layer,
                num_layers=code_transformer_layers,
            )
        else:
            self.code_transformer = None

        if fusion_mode == "concat":
            fusion_input_dim = hidden_dim * 2
        else:
            fusion_input_dim = hidden_dim

        self.fusion_mode = fusion_mode
        self.node_token_interaction = node_token_interaction
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, len(TARGET_NAMES)),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, batch: Batch) -> torch.Tensor:
        # Base graph embedding
        graph_emb = self.graph_encoder.encode_graph(batch)

        # Optional node-token interaction to enrich node states using code tokens
        if self.node_token_interaction:
            node_states = self.graph_encoder.encode_nodes(batch)
            tokens_info = self._get_projected_tokens(batch, device=node_states.device)
            if tokens_info is not None:
                token_proj, seq_lengths, starts, offsets = tokens_info
                num_graphs = int(seq_lengths.size(0))
                fused_nodes = node_states.clone()
                for gidx in range(num_graphs):
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
                graph_emb = global_add_pool(fused_nodes, batch.batch)

        code_emb = self._encode_code(batch, device=graph_emb.device)

        if code_emb is None:
            if self.fusion_mode == "concat":
                zero_pad = torch.zeros_like(graph_emb)
                fused = torch.cat([graph_emb, zero_pad], dim=-1)
            else:
                fused = graph_emb
        elif self.fusion_mode == "concat":
            fused = torch.cat([graph_emb, code_emb], dim=-1)
        elif self.fusion_mode == "add":
            fused = graph_emb + code_emb
        else:
            raise RuntimeError("Unexpected fusion mode")

        return self.fusion_head(fused)

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
            if pooled.dim() == 1:
                pooled = pooled.unsqueeze(0)
            # PyG concatenates along dim=0; ensure batch alignment.
            if pooled.size(0) != num_graphs:
                raise ValueError(
                    "Mismatch between number of graphs and pooled code embeddings."
                )
            projected = self.code_projection(pooled.to(device))
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
        self, batch: Batch, device: torch.device
    ) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Return per-token projected embeddings concatenated across graphs,
        together with per-graph seq lengths and start/end offsets.

        If token embeddings are unavailable, returns None.
        """
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
        projected_all = torch.cat(projected_list, dim=0)
        return projected_all, seq_lengths.to(device), starts.to(device), offsets.to(device)
