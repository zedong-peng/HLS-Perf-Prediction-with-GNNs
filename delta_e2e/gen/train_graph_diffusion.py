#!/usr/bin/env python3
import os
import argparse
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler

try:
    import swanlab
    _SWANLAB_AVAILABLE = True
except ModuleNotFoundError:
    swanlab = None
    _SWANLAB_AVAILABLE = False

try:
    from delta_e2e.gen.data import GraphAttributeDiffusionDataset, pad_collate_fn
    from delta_e2e.gen.model_diffusion import ConditionalGraphDenoiser, GaussianDiffusion
    from delta_e2e.gen.metrics import compute_graph_quality_metrics
except ModuleNotFoundError:
    from data import GraphAttributeDiffusionDataset, pad_collate_fn
    from model_diffusion import ConditionalGraphDenoiser, GaussianDiffusion
    from metrics import compute_graph_quality_metrics


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return F.mse_loss(pred, target)
    mask = mask.to(dtype=pred.dtype, device=pred.device)
    while mask.dim() < pred.dim():
        mask = mask.unsqueeze(-1)
    diff = (pred - target) ** 2 * mask
    denom = mask[..., 0].sum() * pred.size(-1)
    return diff.sum() / denom.clamp_min(1.0)


def main():
    parser = argparse.ArgumentParser(description="Train conditional full-graph diffusion model")

    parser.add_argument("--kernel_base_dir", type=str,
                        default="/home/user/zedongpeng/workspace/HLS-Perf-Prediction-with-GNNs/Graphs/forgehls_kernels/kernels/",
                        help="Kernel root")
    parser.add_argument("--design_base_dir", type=str,
                        default="/home/user/zedongpeng/workspace/Huggingface/forgehls_lite_10designs/",
                        help="Design root")
    parser.add_argument("--cache_root", type=str,
                        default=str(os.path.join(os.path.dirname(__file__), "..", "graph_cache")),
                        help="Graph cache root")
    # OOD root (optional)
    parser.add_argument("--ood_design_base_dir", type=str,
                        default="/home/user/zedongpeng/workspace/Huggingface/forgehls_benchmark",
                        help="OOD Design root (optional; if not exists, OOD eval is skipped)")

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
    parser.add_argument("--node_loss_weight", type=float, default=1.0)
    parser.add_argument("--edge_loss_weight", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training (recommended on GPU)")
    # Sampling/eval controls
    parser.add_argument("--eval_batches", type=int, default=2, help="Num batches to sample for ID/OOD eval per epoch")
    parser.add_argument("--edge_threshold", type=float, default=0.5, help="Threshold for binarizing edges in eval")
    parser.add_argument("--max_nodes", type=int, default=None, help="Optional clamp on per-graph node count (not used here)")
    parser.add_argument("--mode", type=str, default="joint", choices=["joint", "node_latent"], help="Training mode")
    parser.add_argument("--latent_dim", type=int, default=64, help="Node latent dimension for node_latent mode")
    parser.add_argument("--nodes_only", action="store_true", help="Train only node diffusion; skip edge modeling")
    parser.add_argument("--use_swanlab", action="store_true", help="Enable SwanLab logging if available")

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(__file__), "runs", f"graph_diffusion_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    use_swanlab = bool(args.use_swanlab and _SWANLAB_AVAILABLE)
    if use_swanlab:
        try:
            swanlab.init(
                project="HLS-Graph-Diffusion",
                experiment_name=f"GraphDiff_{timestamp}",
                config=vars(args),
                logdir=output_dir,
            )
        except Exception as exc:
            print(f"SwanLab init failed ({exc}); continuing without logging.")
            use_swanlab = False
    elif args.use_swanlab and not _SWANLAB_AVAILABLE:
        print("SwanLab not installed; proceeding without logging.")

    if args.nodes_only and args.mode != "joint":
        raise ValueError("--nodes_only is only supported when mode='joint'")
    if args.nodes_only:
        args.edge_loss_weight = 0.0
        if args.eval_batches > 0:
            print("Running in nodes-only mode; edge metrics will be skipped during evaluation.")

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
        if use_swanlab:
            swanlab.finish()
        return

    # Optional OOD dataset
    ood_dataset = None
    if args.ood_design_base_dir and os.path.exists(args.ood_design_base_dir):
        try:
            ood_dataset = GraphAttributeDiffusionDataset(
                kernel_base_dir=args.kernel_base_dir,
                design_base_dir=args.ood_design_base_dir,
                cache_root=args.cache_root,
                rebuild_cache=False,
                max_pairs=args.max_pairs,
                seed=args.seed + 1,
            )
        except Exception:
            ood_dataset = None
    else:
        print(f"Skip OOD eval: path not found or not provided: {args.ood_design_base_dir}")

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
    id_test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.loader_workers, collate_fn=pad_collate_fn, pin_memory=True)
    ood_loader = None
    if ood_dataset is not None and len(ood_dataset) > 0:
        ood_loader = DataLoader(ood_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.loader_workers, collate_fn=pad_collate_fn, pin_memory=True)

    feature_dim = dataset.feature_dim
    code_feature_dim = dataset.code_feature_dim
    edge_channels = dataset.edge_channels
    diffusion = GaussianDiffusion(timesteps=args.timesteps)
    latent_encoder = None
    if args.mode == "node_latent":
        try:
            from delta_e2e.gen.model_diffusion import NodeLatentEncoder, NodeLatentDenoiser
        except ModuleNotFoundError:
            from model_diffusion import NodeLatentEncoder, NodeLatentDenoiser
        latent_encoder = NodeLatentEncoder(
            feature_dim=feature_dim,
            code_feature_dim=code_feature_dim,
            latent_dim=args.latent_dim,
        ).to(device)
        for p in latent_encoder.parameters():
            p.requires_grad_(False)
        model = NodeLatentDenoiser(
            latent_dim=args.latent_dim,
            feature_dim=feature_dim,
            code_feature_dim=code_feature_dim,
            hidden_dim=args.hidden_dim,
            time_embed_dim=args.time_embed_dim,
        ).to(device)
    else:
        model = ConditionalGraphDenoiser(feature_dim=feature_dim, edge_channels=edge_channels,
                                         hidden_dim=args.hidden_dim, time_embed_dim=args.time_embed_dim,
                                         code_feature_dim=code_feature_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = GradScaler(enabled=amp_enabled)

    # ---------------- Evaluation helper (sampling + metrics) ----------------
    @torch.no_grad()
    def evaluate_generation(sample_loader: DataLoader, prefix: str):
        if sample_loader is None:
            return
        model.eval()
        total = 0
        agg = {
            "node_feature_mae": 0.0,
            "node_feature_rmse": 0.0,
            "node_feature_cosine": 0.0,
            "adjacency_mae": 0.0,
            "adjacency_precision": 0.0,
            "adjacency_recall": 0.0,
            "adjacency_f1": 0.0,
            "degree_l1": 0.0,
            "density_abs": 0.0,
            "edge_attr_mae": 0.0,
            "edge_attr_cosine": 0.0,
        }
        counted = {k: 0 for k in agg.keys()}

        for bi, batch in enumerate(sample_loader):
            if bi >= max(1, args.eval_batches):
                break
            x_cond = batch["x_cond"].to(device)
            code_cond = batch["code_cond"].to(device)
            node_mask = batch["mask"].to(device)
            edge_mask = batch.get("edge_mask")
            if edge_mask is not None:
                edge_mask = edge_mask.to(device)
            x_target = batch["x_target"].to(device)
            edge_target = batch.get("edge_target")
            if edge_target is not None:
                edge_target = edge_target.to(device)

            with autocast(enabled=amp_enabled):
                if args.nodes_only:
                    nodes_gen = diffusion.p_sample_loop(
                        model.node_denoiser,
                        x_cond=x_cond,
                        code_cond=code_cond,
                        mask=node_mask,
                        device=device,
                    )
                    edges_gen = None
                else:
                    nodes_gen, edges_gen = diffusion.p_sample_graph(
                        model, x_cond=x_cond, code_cond=code_cond,
                        node_mask=node_mask, edge_mask=edge_mask, device=device
                    )

            B = x_cond.size(0)
            for b in range(B):
                valid_idx = node_mask[b].nonzero(as_tuple=False).squeeze(-1)
                if valid_idx.numel() == 0:
                    continue
                valid_idx_cpu = valid_idx.detach().cpu()

                # Generated (slice on device, then move to CPU)
                nodes_b = nodes_gen[b, valid_idx].detach().float().cpu()
                if edges_gen is not None:
                    adj_logits_b = edges_gen[b, :, :, 0].detach().float().cpu()
                    adj_probs_b = torch.sigmoid(adj_logits_b)
                    edge_attr_b = edges_gen[b, :, :, 1:].detach().float().cpu()
                else:
                    adj_probs_b = None
                    edge_attr_b = None

                # Targets on CPU
                target_nodes_b = x_target[b, valid_idx].detach().float().cpu()
                if edge_target is not None and edge_mask is not None and not args.nodes_only:
                    target_edge_dense_b = edge_target[b].detach().float().cpu()
                    target_edges_b = target_edge_dense_b.index_select(0, valid_idx_cpu).index_select(1, valid_idx_cpu)[..., 0]
                    target_edge_attr_b = target_edge_dense_b.index_select(0, valid_idx_cpu).index_select(1, valid_idx_cpu)[..., 1:]
                    edge_mask_b = edge_mask[b].detach().cpu().index_select(0, valid_idx_cpu).index_select(1, valid_idx_cpu)
                else:
                    target_edges_b = None
                    target_edge_attr_b = None
                    edge_mask_b = None

                metrics = compute_graph_quality_metrics(
                    nodes_gen=nodes_b,
                    edges_gen=(
                        adj_probs_b.index_select(0, valid_idx_cpu).index_select(1, valid_idx_cpu)
                        if adj_probs_b is not None
                        else None
                    ),
                    edge_attr_gen=(
                        edge_attr_b.index_select(0, valid_idx_cpu).index_select(1, valid_idx_cpu)
                        if edge_attr_b is not None
                        else None
                    ),
                    target_nodes=target_nodes_b,
                    target_edges=target_edges_b,
                    target_edge_attr=target_edge_attr_b,
                    node_mask=torch.ones_like(target_nodes_b[:, 0], dtype=torch.bool),
                    edge_mask=edge_mask_b,
                    threshold=float(args.edge_threshold),
                )
                # Aggregate
                for k, v in metrics.items():
                    if v is not None:
                        agg[k] += float(v)
                        counted[k] += 1
                total += 1

        if total == 0:
            return
        # Averages
        averaged = {k: (agg[k] / max(1, counted[k])) if counted[k] > 0 else None for k in agg.keys()}
        # Log
        log_payload = {f"{prefix}/{k}": (float(v) if v is not None else 0.0) for k, v in averaged.items()}
        log_payload[f"{prefix}/evaluated_samples"] = int(total)
        if use_swanlab:
            swanlab.log(log_payload)

    # ----------------------------------------------------------------------

    global_step = 0
    best_valid = float("inf")
    best_path = os.path.join(output_dir, "best_graph_diffusion.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_node_loss = 0.0
        epoch_edge_loss = 0.0
        count = 0
        for batch in train_loader:
            x0 = batch["x_target"].to(device)
            x_cond = batch["x_cond"].to(device)
            code_cond = batch["code_cond"].to(device)
            node_mask = batch["mask"].to(device)

            B = x0.size(0)
            t = torch.randint(0, diffusion.timesteps, (B,), device=device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)
            if args.mode == "node_latent":
                with torch.no_grad():
                    z0 = latent_encoder(x0, x_cond, code_cond, node_mask)
                z_t, z_noise = diffusion.q_sample(z0, t)
                with autocast(enabled=amp_enabled):
                    eps_pred = model(z_t, x_cond, code_cond, t, node_mask)
                    loss = masked_mse(eps_pred, z_noise, node_mask)
                node_loss = loss
                edge_loss = torch.zeros((), device=device)
            else:
                if args.nodes_only:
                    x_t, node_noise = diffusion.q_sample(x0, t)
                    with autocast(enabled=amp_enabled):
                        node_eps = model.node_denoiser(x_t, x_cond, code_cond, t, node_mask)
                        node_loss = masked_mse(node_eps, node_noise, node_mask)
                        edge_loss = torch.zeros((), device=device)
                        loss = args.node_loss_weight * node_loss
                else:
                    if args.nodes_only:
                        x_t, node_noise = diffusion.q_sample(x0, t)
                        with autocast(enabled=amp_enabled):
                            node_eps = model.node_denoiser(x_t, x_cond, code_cond, t, node_mask)
                            node_loss = masked_mse(node_eps, node_noise, node_mask)
                            edge_loss = torch.zeros((), device=device)
                    else:
                        edge0 = batch["edge_target"].to(device)
                        edge_mask = batch["edge_mask"].to(device)
                        x_t, node_noise = diffusion.q_sample(x0, t)
                        edge_t, edge_noise = diffusion.q_sample(edge0, t)
                        with autocast(enabled=amp_enabled):
                            outputs = model(x_t, edge_t, x_cond, code_cond, t, node_mask, edge_mask)
                            node_loss = masked_mse(outputs["node"], node_noise, node_mask)
                            edge_loss = masked_mse(outputs["edge"], edge_noise, edge_mask)
                        loss = args.node_loss_weight * node_loss + args.edge_loss_weight * edge_loss

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_node_loss += node_loss.item()
            epoch_edge_loss += edge_loss.item()
            count += 1
            global_step += 1

            if use_swanlab and global_step % 50 == 0:
                swanlab.log({
                    "train/node_loss": node_loss.item(),
                    "train/edge_loss": edge_loss.item(),
                    "train/total_loss": loss.item(),
                    "step": global_step,
                })

        avg_node_train = epoch_node_loss / max(1, count)
        avg_edge_train = epoch_edge_loss / max(1, count)
        avg_train = args.node_loss_weight * avg_node_train + args.edge_loss_weight * avg_edge_train

        model.eval()
        with torch.no_grad():
            val_node = 0.0
            val_edge = 0.0
            v_count = 0
            for batch in valid_loader:
                x0 = batch["x_target"].to(device)
                x_cond = batch["x_cond"].to(device)
                code_cond = batch["code_cond"].to(device)
                node_mask = batch["mask"].to(device)
                B = x0.size(0)
                t = torch.randint(0, diffusion.timesteps, (B,), device=device, dtype=torch.long)
                if args.mode == "node_latent":
                    with torch.no_grad():
                        z0 = latent_encoder(x0, x_cond, code_cond, node_mask)
                    z_t, z_noise = diffusion.q_sample(z0, t)
                    with autocast(enabled=amp_enabled):
                        eps_pred = model(z_t, x_cond, code_cond, t, node_mask)
                        node_loss = masked_mse(eps_pred, z_noise, node_mask)
                        edge_loss = torch.zeros((), device=device)
                else:
                    if args.nodes_only:
                        x_t, node_noise = diffusion.q_sample(x0, t)
                        with autocast(enabled=amp_enabled):
                            node_eps = model.node_denoiser(x_t, x_cond, code_cond, t, node_mask)
                            node_loss = masked_mse(node_eps, node_noise, node_mask)
                            edge_loss = torch.zeros((), device=device)
                    else:
                        edge0 = batch["edge_target"].to(device)
                        edge_mask = batch["edge_mask"].to(device)
                        x_t, node_noise = diffusion.q_sample(x0, t)
                        edge_t, edge_noise = diffusion.q_sample(edge0, t)
                        with autocast(enabled=amp_enabled):
                            outputs = model(x_t, edge_t, x_cond, code_cond, t, node_mask, edge_mask)
                            node_loss = masked_mse(outputs["node"], node_noise, node_mask)
                            edge_loss = masked_mse(outputs["edge"], edge_noise, edge_mask)
                val_node += float(node_loss.item())
                val_edge += float(edge_loss.item()) if edge_loss.numel() > 0 else 0.0
                v_count += 1
            avg_node_valid = val_node / max(1, v_count)
            avg_edge_valid = val_edge / max(1, v_count)
            avg_valid = args.node_loss_weight * avg_node_valid + args.edge_loss_weight * avg_edge_valid

        if use_swanlab:
            swanlab.log({
                "epoch": epoch,
                "train/node_loss_avg": avg_node_train,
                "train/edge_loss_avg": avg_edge_train,
                "train/total_loss_avg": avg_train,
                "valid/node_loss_avg": avg_node_valid,
                "valid/edge_loss_avg": avg_edge_valid,
                "valid/total_loss_avg": avg_valid,
            })

        # Sampling-based evaluation on ID test and OOD (if available)
        if args.mode != "node_latent" and int(args.eval_batches) > 0:
            evaluate_generation(id_test_loader, prefix="id_eval")
            if ood_loader is not None:
                evaluate_generation(ood_loader, prefix="ood_eval")

        if avg_valid < best_valid:
            best_valid = avg_valid
            torch.save({
                "model_state": model.state_dict(),
                "feature_dim": feature_dim,
                "hidden_dim": args.hidden_dim,
                "time_embed_dim": args.time_embed_dim,
                "timesteps": args.timesteps,
                "code_feature_dim": code_feature_dim,
                "edge_channels": edge_channels,
                "node_loss_weight": args.node_loss_weight,
                "edge_loss_weight": args.edge_loss_weight,
            }, best_path)
            if use_swanlab:
                swanlab.log({"best/epoch": epoch, "best/valid_loss": best_valid})

    if use_swanlab:
        swanlab.finish()
    print(f"Training finished. Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
