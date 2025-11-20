#!/usr/bin/env python3
import os
import json
import argparse
from datetime import datetime

import torch

from delta_e2e.gen.data import GraphAttributeDiffusionDataset
from delta_e2e.gen.inference import load_diffusion_bundle, sample_conditioned_graph
from delta_e2e.gen.metrics import compute_graph_quality_metrics


def main():
    parser = argparse.ArgumentParser(description="Sample full design graphs from the diffusion model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--kernel_base_dir", type=str,
                        default="/home/user/zedongpeng/workspace/HLS-Perf-Prediction-with-GNNs/Graphs/forgehls_kernels/kernels/")
    parser.add_argument("--design_base_dir", type=str,
                        default="/home/user/zedongpeng/workspace/Huggingface/forgehls_lite_10designs/")
    parser.add_argument("--cache_root", type=str,
                        default=str(os.path.join(os.path.dirname(__file__), "..", "graph_cache")))
    parser.add_argument("--pair_index", type=int, default=0, help="Index of pair to condition on")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default=os.path.join(os.path.dirname(__file__), "samples"))
    parser.add_argument("--edge_threshold", type=float, default=0.5, help="Threshold on sigmoid(adjacency)")
    parser.add_argument("--include_target", action="store_true", help="Export ground-truth node/edge tensors")
    parser.add_argument("--nodes_only", action="store_true", help="Sample only node features and skip edge generation")

    args = parser.parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.out_dir, exist_ok=True)

    bundle = load_diffusion_bundle(args.checkpoint, device)

    dataset = GraphAttributeDiffusionDataset(
        kernel_base_dir=args.kernel_base_dir,
        design_base_dir=args.design_base_dir,
        cache_root=args.cache_root,
        rebuild_cache=False,
        max_pairs=None,
        seed=123,
    )

    if len(dataset) == 0:
        print("No usable pairs for sampling.")
        return

    index = max(0, min(args.pair_index, len(dataset) - 1))
    sample = dataset[index]

    generated = sample_conditioned_graph(
        bundle,
        sample,
        device,
        edge_threshold=args.edge_threshold,
        nodes_only=args.nodes_only,
    )
    nodes_gen = generated.node_features
    pair_mask = generated.edge_mask

    result_generated = {
        "node_features": nodes_gen.tolist(),
    }

    if (
        generated.adj_logits is not None
        and generated.adj_probs is not None
        and generated.adj_binary is not None
        and generated.edge_attr is not None
    ):
        adj_logits = generated.adj_logits
        adj_probs = generated.adj_probs
        adj_binary = generated.adj_binary
        edge_attr_pred = generated.edge_attr

        edge_indices = generated.edge_index().tolist()
        edge_attr_list = generated.edge_attr_list().tolist()

        result_generated.update(
            {
                "adjacency_logits": adj_logits.tolist(),
                "adjacency_probs": adj_probs.tolist(),
                "adjacency_binary": adj_binary.tolist(),
                "edge_attr_dense": edge_attr_pred.tolist(),
                "edge_index": edge_indices,
                "edge_attr": edge_attr_list,
            }
        )
    else:
        adj_logits = None
        adj_probs = None
        adj_binary = None
        edge_attr_pred = None

    result = {
        "pair_id": sample["pair_id"],
        "generated": result_generated,
        "conditioning": {
            "x_cond": sample["x_cond"].tolist(),
            "code_cond": sample["code_cond"].tolist(),
            "node_mask": sample["mask"].tolist(),
            "edge_mask": pair_mask.tolist(),
        },
    }

    if args.include_target:
        result["target"] = {
            "node_features": sample["x_target"].tolist(),
            "edge_target": sample["edge_target"].tolist(),
        }

    metrics = compute_graph_quality_metrics(
        nodes_gen=nodes_gen,
        edges_gen=adj_probs,
        edge_attr_gen=edge_attr_pred,
        target_nodes=sample["x_target"],
        target_edges=sample["edge_target"][..., 0] if not args.nodes_only else None,
        target_edge_attr=sample["edge_target"][..., 1:] if not args.nodes_only else None,
        node_mask=sample["mask"],
        edge_mask=pair_mask if not args.nodes_only else None,
    )
    result["quality_metrics"] = metrics

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out_dir, f"gen_graph_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved generated graph to: {out_path}")


if __name__ == "__main__":
    main()
