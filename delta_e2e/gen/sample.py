#!/usr/bin/env python3
import os
import json
import argparse
from datetime import datetime

import torch

from delta_e2e.gen.data import GraphAttributeDiffusionDataset, pad_collate_fn
from delta_e2e.gen.model_diffusion import ConditionalNodeDenoiser, GaussianDiffusion


def main():
    parser = argparse.ArgumentParser(description="Sample from conditional attribute diffusion model")
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

    args = parser.parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    feature_dim = int(ckpt["feature_dim"])
    hidden_dim = int(ckpt["hidden_dim"])
    time_embed_dim = int(ckpt["time_embed_dim"])
    timesteps = int(ckpt["timesteps"])
    code_feature_dim = int(ckpt.get("code_feature_dim", 16))

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

    x_cond = sample["x_cond"].unsqueeze(0).to(device)    # [1, N, F]
    code_cond = sample["code_cond"].unsqueeze(0).to(device)  # [1, N, C]
    mask = sample["mask"].unsqueeze(0).to(device)        # [1, N]

    model = ConditionalNodeDenoiser(feature_dim=feature_dim, hidden_dim=hidden_dim,
                                    time_embed_dim=time_embed_dim, code_feature_dim=code_feature_dim).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    diffusion = GaussianDiffusion(timesteps=timesteps)

    with torch.no_grad():
        x_gen = diffusion.p_sample_loop(model, x_cond=x_cond, code_cond=code_cond, mask=mask, device=device)  # [1, N, F]

    x_gen = x_gen.squeeze(0).cpu().numpy().tolist()

    out = {
        "pair_id": sample.get("pair_id", f"pair_{index}"),
        "generated_node_features": x_gen,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out_dir, f"gen_attr_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Saved generated node features to: {out_path}")


if __name__ == "__main__":
    main() 