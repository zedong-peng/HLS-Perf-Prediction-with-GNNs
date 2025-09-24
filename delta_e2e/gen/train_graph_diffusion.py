#!/usr/bin/env python3
import os
import argparse
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import swanlab

from delta_e2e.gen.data import GraphAttributeDiffusionDataset, pad_collate_fn
from delta_e2e.gen.model_diffusion import ConditionalNodeDenoiser, GaussianDiffusion


def main():
    parser = argparse.ArgumentParser(description="Train conditional graph attribute diffusion (PoC)")

    parser.add_argument("--kernel_base_dir", type=str,
                        default="/home/user/zedongpeng/workspace/HLS-Perf-Prediction-with-GNNs/Graphs/forgehls_kernels/kernels/",
                        help="Kernel root")
    parser.add_argument("--design_base_dir", type=str,
                        default="/home/user/zedongpeng/workspace/Huggingface/forgehls_lite_10designs/",
                        help="Design root")
    parser.add_argument("--cache_root", type=str,
                        default=str(os.path.join(os.path.dirname(__file__), "..", "graph_cache")),
                        help="Graph cache root")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--time_embed_dim", type=int, default=128)
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loader_workers", type=int, default=4)

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(__file__), "runs", f"attr_diffusion_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    swanlab.init(
        project="HLS-Graph-Diffusion",
        experiment_name=f"AttrDiff_PoC_{timestamp}",
        config=vars(args),
        logdir=output_dir,
    )

    dataset = GraphAttributeDiffusionDataset(
        kernel_base_dir=args.kernel_base_dir,
        design_base_dir=args.design_base_dir,
        cache_root=args.cache_root,
        rebuild_cache=False,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )

    if len(dataset) == 0 or dataset.feature_dim == 0:
        print("No valid pairs for training.")
        swanlab.finish()
        return

    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    valid_len = int(0.1 * total_len)
    test_len = total_len - train_len - valid_len

    generator = torch.Generator().manual_seed(args.seed)
    train_ds, valid_ds, test_ds = random_split(dataset, [train_len, valid_len, test_len], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.loader_workers, collate_fn=pad_collate_fn, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.loader_workers, collate_fn=pad_collate_fn, pin_memory=True)

    feature_dim = dataset.feature_dim
    code_feature_dim = dataset.code_feature_dim
    model = ConditionalNodeDenoiser(feature_dim=feature_dim, hidden_dim=args.hidden_dim,
                                    time_embed_dim=args.time_embed_dim, code_feature_dim=code_feature_dim).to(device)
    diffusion = GaussianDiffusion(timesteps=args.timesteps)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    global_step = 0
    best_valid = float("inf")
    best_path = os.path.join(output_dir, "best_attr_diffusion.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        count = 0
        for batch in train_loader:
            x0 = batch["x_target"].to(device)
            x_cond = batch["x_cond"].to(device)
            code_cond = batch["code_cond"].to(device)
            mask = batch["mask"].to(device)

            B = x0.size(0)
            t = torch.randint(0, diffusion.timesteps, (B,), device=device, dtype=torch.long)
            x_t, noise = diffusion.q_sample(x0, t)

            eps_pred = model(x_t, x_cond, code_cond, t, mask)
            loss = F.mse_loss(eps_pred[mask], noise[mask])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            count += 1
            global_step += 1

            if global_step % 50 == 0:
                swanlab.log({"train/loss": loss.item(), "step": global_step})

        avg_train = epoch_loss / max(1, count)

        # validation
        model.eval()
        with torch.no_grad():
            v_loss = 0.0
            v_count = 0
            for batch in valid_loader:
                x0 = batch["x_target"].to(device)
                x_cond = batch["x_cond"].to(device)
                code_cond = batch["code_cond"].to(device)
                mask = batch["mask"].to(device)
                B = x0.size(0)
                t = torch.randint(0, diffusion.timesteps, (B,), device=device, dtype=torch.long)
                x_t, noise = diffusion.q_sample(x0, t)
                eps_pred = model(x_t, x_cond, code_cond, t, mask)
                loss = F.mse_loss(eps_pred[mask], noise[mask])
                v_loss += loss.item()
                v_count += 1
            avg_valid = v_loss / max(1, v_count)

        swanlab.log({"epoch": epoch, "train/avg_loss": avg_train, "valid/avg_loss": avg_valid})

        if avg_valid < best_valid:
            best_valid = avg_valid
            torch.save({
                "model_state": model.state_dict(),
                "feature_dim": feature_dim,
                "hidden_dim": args.hidden_dim,
                "time_embed_dim": args.time_embed_dim,
                "timesteps": args.timesteps,
                "code_feature_dim": code_feature_dim,
            }, best_path)
            swanlab.log({"best/epoch": epoch, "best/valid_loss": best_valid})

    swanlab.finish()
    print(f"Training finished. Best checkpoint: {best_path}")


if __name__ == "__main__":
    main() 