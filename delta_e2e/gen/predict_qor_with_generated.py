#!/usr/bin/env python3
import os
import json
import argparse
from datetime import datetime

import torch
from torch_geometric.data import Data

from delta_e2e.gen.data import GraphAttributeDiffusionDataset
from delta_e2e.gen.inference import load_diffusion_bundle, sample_conditioned_graph
from delta_e2e.gen.metrics import compute_graph_quality_metrics
from delta_e2e.train_e2e import SimpleDifferentialGNN

METRIC_IDX = {'dsp': 0, 'lut': 1, 'ff': 2, 'latency': 3}


def load_qor_model(ckpt_path: str, node_dim: int, device: torch.device) -> SimpleDifferentialGNN:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    args = ckpt['args']
    differential = str(args.get('differential', 'true')).lower() == 'true'
    model = SimpleDifferentialGNN(
        node_dim=node_dim,
        hidden_dim=args['hidden_dim'],
        num_layers=args['num_layers'],
        dropout=args['dropout'],
        target_metric=args['target_metric'],
        differential=differential,
        gnn_type=args.get('gnn_type', 'gcn')
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, args


def build_design_graph(sample: dict, generated, device: torch.device) -> Data:
    node_mask = sample['mask']
    valid_idx = node_mask.nonzero(as_tuple=False).squeeze(-1)
    node_features = generated.node_features[valid_idx]

    adj_binary = generated.adj_binary.index_select(0, valid_idx).index_select(1, valid_idx)
    edge_attr_dense = generated.edge_attr.index_select(0, valid_idx).index_select(1, valid_idx)
    edge_mask = generated.edge_mask.index_select(0, valid_idx).index_select(1, valid_idx)

    edge_positions = adj_binary.nonzero(as_tuple=False)
    attr_dim = edge_attr_dense.size(-1)
    if edge_positions.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, attr_dim), dtype=edge_attr_dense.dtype)
    else:
        edge_index = edge_positions.t().contiguous()
        edge_attr = edge_attr_dense[edge_positions[:, 0], edge_positions[:, 1]]

    data = Data(
        x=node_features.to(device=device, dtype=torch.float32),
        edge_index=edge_index.to(device),
        edge_attr=edge_attr.to(device) if attr_dim > 0 else None,
        y=sample['design_graph'].y.clone().to(device),
    )
    data.edge_mask = edge_mask  # keep for metrics inspection if needed
    return data, valid_idx


def main():
    parser = argparse.ArgumentParser(description="Generate design graph and predict QoR using trained models")
    parser.add_argument("--diffusion_ckpt", type=str, required=True)
    parser.add_argument("--qor_ckpt", type=str, required=True)
    parser.add_argument("--pair_index", type=int, default=0)
    parser.add_argument("--edge_threshold", type=float, default=0.5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--kernel_base_dir", type=str,
                        default="/home/user/zedongpeng/workspace/HLS-Perf-Prediction-with-GNNs/Graphs/forgehls_kernels/kernels/")
    parser.add_argument("--design_base_dir", type=str,
                        default="/home/user/zedongpeng/workspace/Huggingface/forgehls_lite_10designs/")
    parser.add_argument("--cache_root", type=str,
                        default=str(os.path.join(os.path.dirname(__file__), "..", "graph_cache")))
    parser.add_argument("--out_dir", type=str, default=os.path.join(os.path.dirname(__file__), "runs"))

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    bundle = load_diffusion_bundle(args.diffusion_ckpt, device)
    dataset = GraphAttributeDiffusionDataset(
        kernel_base_dir=args.kernel_base_dir,
        design_base_dir=args.design_base_dir,
        cache_root=args.cache_root,
        rebuild_cache=False,
        max_pairs=None,
        seed=123,
    )

    if len(dataset) == 0:
        raise RuntimeError("No data available for QoR prediction.")

    idx = max(0, min(args.pair_index, len(dataset) - 1))
    sample = dataset[idx]

    generated = sample_conditioned_graph(bundle, sample, device, edge_threshold=args.edge_threshold)
    design_graph_gen, valid_idx = build_design_graph(sample, generated, device)

    kernel_graph = sample['kernel_graph'].clone()
    kernel_graph = kernel_graph.to(device)

    qor_model, qor_args = load_qor_model(args.qor_ckpt, node_dim=kernel_graph.x.size(1), device=device)

    pragma_count = sample['pragma_count'].to(device)
    kernel_graph.batch = torch.zeros(kernel_graph.num_nodes, dtype=torch.long, device=device)
    design_graph_gen.batch = torch.zeros(design_graph_gen.num_nodes, dtype=torch.long, device=device)

    with torch.no_grad():
        output = qor_model(kernel_graph, design_graph_gen, pragma_count)

    target_metric = qor_args['target_metric']
    metric_idx = METRIC_IDX[target_metric]
    kernel_metric = float(sample['kernel_graph'].y[0, metric_idx].item())
    design_metric_true = float(sample['design_graph'].y[0, metric_idx].item())

    if qor_model.differential:
        predicted_delta = float(output['delta_pred'].item())
        design_metric_pred = kernel_metric + predicted_delta
    else:
        predicted_delta = None
        design_metric_pred = float(output['direct_pred'].item())

    node_mask = sample['mask']
    target_nodes = sample['x_target'][valid_idx]
    target_edge_dense = sample['edge_target']
    target_edges = target_edge_dense.index_select(0, valid_idx).index_select(1, valid_idx)[..., 0]
    target_edge_attr = target_edge_dense.index_select(0, valid_idx).index_select(1, valid_idx)[..., 1:]
    edge_mask = sample['edge_mask'].index_select(0, valid_idx).index_select(1, valid_idx)

    metrics = compute_graph_quality_metrics(
        nodes_gen=generated.node_features.index_select(0, valid_idx),
        edges_gen=generated.adj_probs.index_select(0, valid_idx).index_select(1, valid_idx),
        edge_attr_gen=generated.edge_attr.index_select(0, valid_idx).index_select(1, valid_idx),
        target_nodes=target_nodes,
        target_edges=target_edges,
        target_edge_attr=target_edge_attr,
        node_mask=torch.ones_like(target_nodes[:, 0], dtype=torch.bool),
        edge_mask=edge_mask,
        threshold=args.edge_threshold,
    )

    result = {
        "pair_id": sample['pair_id'],
        "target_metric": target_metric,
        "kernel_metric": kernel_metric,
        "design_metric_true": design_metric_true,
        "design_metric_pred": design_metric_pred,
        "predicted_delta": predicted_delta,
        "edge_threshold": args.edge_threshold,
        "num_nodes_pred": int(design_graph_gen.num_nodes),
        "num_edges_pred": int(design_graph_gen.edge_index.size(1)),
        "graph_quality": metrics,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out_dir, f"qor_eval_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"Saved QoR prediction summary to: {out_path}")


if __name__ == "__main__":
    main()
