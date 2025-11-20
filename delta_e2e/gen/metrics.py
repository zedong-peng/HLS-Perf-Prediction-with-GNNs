from typing import Dict, Optional

import torch
import torch.nn.functional as F


def compute_graph_quality_metrics(
    nodes_gen: torch.Tensor,
    edges_gen: Optional[torch.Tensor],
    edge_attr_gen: Optional[torch.Tensor],
    target_nodes: torch.Tensor,
    target_edges: Optional[torch.Tensor],
    target_edge_attr: Optional[torch.Tensor],
    node_mask: torch.Tensor,
    edge_mask: Optional[torch.Tensor],
    threshold: float = 0.5,
) -> Dict[str, Optional[float]]:
    device = torch.device("cpu")
    nodes_gen = nodes_gen.to(device=device, dtype=torch.float32)
    target_nodes = target_nodes.to(device=device, dtype=torch.float32)
    node_mask = node_mask.to(device=device, dtype=torch.bool)

    metrics: Dict[str, Optional[float]] = {}

    if node_mask.any():
        masked_nodes_gen = nodes_gen[node_mask]
        masked_nodes_target = target_nodes[node_mask]
        diff = (masked_nodes_gen - masked_nodes_target).abs()
        metrics["node_feature_mae"] = float(diff.mean().item())
        rmse = torch.sqrt(((masked_nodes_gen - masked_nodes_target) ** 2).mean())
        metrics["node_feature_rmse"] = float(rmse.item())
        cosine = F.cosine_similarity(masked_nodes_gen, masked_nodes_target, dim=-1)
        metrics["node_feature_cosine"] = float(cosine.mean().item())
    else:
        metrics["node_feature_mae"] = None
        metrics["node_feature_rmse"] = None
        metrics["node_feature_cosine"] = None
    has_edge_inputs = edges_gen is not None and target_edges is not None and edge_mask is not None

    if has_edge_inputs:
        edges_gen = edges_gen.to(device=device, dtype=torch.float32)
        target_edges = target_edges.to(device=device, dtype=torch.float32)
        edge_mask = edge_mask.to(device=device, dtype=torch.bool)
        if edge_attr_gen is not None and target_edge_attr is not None:
            edge_attr_gen = edge_attr_gen.to(device=device, dtype=torch.float32)
            target_edge_attr = target_edge_attr.to(device=device, dtype=torch.float32)
        else:
            edge_attr_gen = None
            target_edge_attr = None

        n = edges_gen.size(0)
        diag_mask = torch.eye(n, dtype=torch.bool)
        valid_pairs = edge_mask & ~diag_mask
        if valid_pairs.any():
            probs = edges_gen * valid_pairs.float()
            target = target_edges * valid_pairs.float()
            metrics["adjacency_mae"] = float((probs - target).abs().sum() / valid_pairs.sum().clamp_min(1).float())

            pred_binary = (edges_gen >= threshold).bool() & valid_pairs
            target_binary = (target_edges >= 0.5).bool() & valid_pairs

            tp = (pred_binary & target_binary).sum().float()
            fp = (pred_binary & ~target_binary).sum().float()
            fn = (~pred_binary & target_binary).sum().float()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            metrics["adjacency_precision"] = float(precision.item())
            metrics["adjacency_recall"] = float(recall.item())
            metrics["adjacency_f1"] = float(f1.item())

            degree_pred = pred_binary.float().sum(dim=1)
            degree_target = target_binary.float().sum(dim=1)
            metrics["degree_l1"] = float((degree_pred - degree_target).abs().mean().item())

            density_pred = degree_pred.sum() / valid_pairs.sum().float()
            density_target = degree_target.sum() / valid_pairs.sum().float()
            metrics["density_abs"] = float(torch.abs(density_pred - density_target).item())

            if edge_attr_gen is not None and target_edge_attr is not None and edge_attr_gen.size(-1) > 0 and target_edge_attr.size(-1) > 0:
                attr_mask = target_binary
                if attr_mask.any():
                    attr_mask_exp = attr_mask.unsqueeze(-1)
                    attr_diff = (edge_attr_gen - target_edge_attr).abs() * attr_mask_exp.float()
                    denom = attr_mask.sum().float() * edge_attr_gen.size(-1)
                    metrics["edge_attr_mae"] = float(attr_diff.sum() / denom.clamp_min(1.0))

                    pred_attr = edge_attr_gen[attr_mask]
                    target_attr = target_edge_attr[attr_mask]
                    cosine = F.cosine_similarity(pred_attr, target_attr, dim=-1)
                    metrics["edge_attr_cosine"] = float(cosine.mean().item())
                else:
                    metrics["edge_attr_mae"] = None
                    metrics["edge_attr_cosine"] = None
            else:
                metrics["edge_attr_mae"] = None
                metrics["edge_attr_cosine"] = None
        else:
            metrics["adjacency_mae"] = None
            metrics["adjacency_precision"] = None
            metrics["adjacency_recall"] = None
            metrics["adjacency_f1"] = None
            metrics["degree_l1"] = None
            metrics["density_abs"] = None
            metrics["edge_attr_mae"] = None
            metrics["edge_attr_cosine"] = None
    else:
        metrics["adjacency_mae"] = None
        metrics["adjacency_precision"] = None
        metrics["adjacency_recall"] = None
        metrics["adjacency_f1"] = None
        metrics["degree_l1"] = None
        metrics["density_abs"] = None
        metrics["edge_attr_mae"] = None
        metrics["edge_attr_cosine"] = None

    return metrics
