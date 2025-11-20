#!/usr/bin/env python3
"""End-to-end QoR prediction using design graphs only."""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:  # pragma: no cover - optional dependency
    import swanlab
    _SWANLAB_AVAILABLE = True
except ImportError:  # pragma: no cover - graceful fallback
    class _SwanLabStub:
        def init(self, *_, **__):
            return None

        def log(self, *_, **__):
            return None

        def finish(self, *_, **__):
            return None

    swanlab = _SwanLabStub()  # type: ignore
    _SWANLAB_AVAILABLE = False

from dataset import DesignQoRDataset, QoRPairRecord, _collect_pair_records
from metrics import compute_regression_metrics
from model import DesignQoRGNN

METRIC_NAMES = ["dsp", "lut", "ff", "latency"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end design QoR prediction")

    parser.add_argument("--kernel_base_dir", type=str,
                        default="/home/user/zedongpeng/workspace/HLS-Perf-Prediction-with-GNNs/Graphs/forgehls_kernels/kernels",
                        help="Kernel graph root")
    parser.add_argument("--design_base_dir", type=str,
                        default="/home/user/zedongpeng/workspace/Huggingface/forgehls_lite_100designs",
                        help="Design dataset root (ID)")
    parser.add_argument("--ood_design_base_dir", type=str,
                        default="/home/user/zedongpeng/workspace/Huggingface/forgehls_benchmark",
                        help="OOD design dataset root")
    parser.add_argument("--cache_root", type=str,
                        default="/home/user/zedongpeng/workspace/HLS-Perf-Prediction-with-GNNs/delta_e2e/graph_cache",
                        help="Shared cache directory (reused from delta_e2e)")
    parser.add_argument("--output_dir", type=str,
                        default=str(CURRENT_DIR / "output" / "forgehls_100designs"),
                        help="Output directory for checkpoints and metrics")

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gnn_type", type=str, default="gcn",
                        choices=["gcn", "gin", "rgcn", "fast_rgcn"])
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_pairs", type=int, default=0)
    parser.add_argument("--rebuild_cache", action="store_true")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_swanlab", action="store_true")

    return parser.parse_args()


def split_records(
    records: Sequence[QoRPairRecord],
    train_ratio: float,
    valid_ratio: float,
    seed: int,
) -> Dict[str, List[QoRPairRecord]]:
    total = len(records)
    if total == 0:
        return {"train": [], "valid": [], "test": []}

    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)
    shuffled = [records[i] for i in indices]

    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)

    train_records = shuffled[:train_end] if train_end > 0 else []
    valid_records = shuffled[train_end:valid_end] if valid_end > train_end else []
    test_records = shuffled[valid_end:] if valid_end < total else []

    # Ensure we have at least one batch for validation/test when data is small.
    if not valid_records and len(train_records) > 1:
        valid_records.append(train_records.pop())
    if not test_records and len(train_records) > 1:
        test_records.append(train_records.pop())

    return {
        "train": train_records,
        "valid": valid_records,
        "test": test_records,
    }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    graphs = [item["design_graph"] for item in batch]
    metrics = torch.stack([item["design_metrics"] for item in batch])
    pair_ids = [item["pair_id"] for item in batch]
    graph_batch = Batch.from_data_list(graphs)
    return {
        "design_graph": graph_batch,
        "design_metrics": metrics,
        "pair_ids": pair_ids,
    }


def train_epoch(
    model: DesignQoRGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_accum_steps: int,
    max_grad_norm: float,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_samples = 0

    for step, batch in enumerate(loader):
        graph_batch = batch["design_graph"].to(device, non_blocking=True)
        targets = batch["design_metrics"].to(device, non_blocking=True)

        preds = model(graph_batch)
        loss = F.l1_loss(preds, targets)
        (loss / max(1, grad_accum_steps)).backward()

        should_step = ((step + 1) % max(1, grad_accum_steps) == 0) or (step + 1 == len(loader))
        if should_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * targets.size(0)
        total_samples += targets.size(0)

    return total_loss / max(1, total_samples)


@torch.no_grad()
def evaluate(
    model: DesignQoRGNN,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, float], torch.Tensor, torch.Tensor]:
    model.eval()
    preds: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []

    for batch in loader:
        graph_batch = batch["design_graph"].to(device, non_blocking=True)
        target = batch["design_metrics"].to(device, non_blocking=True)
        pred = model(graph_batch)
        preds.append(pred.cpu())
        targets.append(target.cpu())

    if not preds:
        empty = torch.empty(0, len(METRIC_NAMES))
        return {}, empty, empty

    pred_tensor = torch.cat(preds, dim=0)
    target_tensor = torch.cat(targets, dim=0)
    metrics = compute_regression_metrics(pred_tensor, target_tensor)
    return metrics, pred_tensor, target_tensor


def collect_predictions(
    model: DesignQoRGNN,
    loader: DataLoader,
    device: torch.device,
) -> List[Dict[str, float]]:
    model.eval()
    results: List[Dict[str, float]] = []

    with torch.no_grad():
        for batch in loader:
            graph_batch = batch["design_graph"].to(device, non_blocking=True)
            preds = model(graph_batch).cpu()
            targets = batch["design_metrics"]
            pair_ids = batch["pair_ids"]

            for pid, pred, target in zip(pair_ids, preds, targets):
                entry = {"pair_id": pid}
                for idx, name in enumerate(METRIC_NAMES):
                    entry[f"pred_{name}"] = float(pred[idx].item())
                    entry[f"true_{name}"] = float(target[idx].item())
                results.append(entry)
    return results


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.use_swanlab and _SWANLAB_AVAILABLE:
        swanlab.init(
            project="HLS-QoR-End2End",
            experiment_name=Path(args.output_dir).name,
            config=vars(args),
            logdir=args.output_dir,
        )

    max_pairs = None if args.max_pairs <= 0 else args.max_pairs
    id_records = _collect_pair_records(
        kernel_base_dir=args.kernel_base_dir,
        design_base_dir=args.design_base_dir,
        cache_root=args.cache_root,
        rebuild_cache=args.rebuild_cache,
        max_pairs=max_pairs,
        seed=args.seed,
    )
    if not id_records:
        raise RuntimeError("No design pairs found. Check dataset paths or rebuild the cache.")

    sample_payload = torch.load(id_records[0].file_path, map_location="cpu")
    node_dim = int(sample_payload["design_graph"].x.size(1))

    splits = split_records(id_records, args.train_ratio, args.valid_ratio, args.seed)
    train_ds = DesignQoRDataset(splits["train"])
    valid_ds = DesignQoRDataset(splits["valid"])
    test_ds = DesignQoRDataset(splits["test"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_fn)

    ood_records: List[QoRPairRecord] = []
    ood_loader: Optional[DataLoader] = None
    if args.ood_design_base_dir and os.path.exists(args.ood_design_base_dir):
        ood_records = _collect_pair_records(
            kernel_base_dir=args.kernel_base_dir,
            design_base_dir=args.ood_design_base_dir,
            cache_root=args.cache_root,
            rebuild_cache=args.rebuild_cache,
            max_pairs=max_pairs,
            seed=args.seed + 1,
        )
        if ood_records:
            ood_ds = DesignQoRDataset(ood_records)
            ood_loader = DataLoader(ood_ds, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, collate_fn=collate_fn)

    model = DesignQoRGNN(
        node_dim=node_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        gnn_type=args.gnn_type,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = math.inf
    best_state: Optional[Dict[str, torch.Tensor]] = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            grad_accum_steps=args.grad_accum_steps,
            max_grad_norm=args.max_grad_norm,
        )

        val_metrics, _, _ = evaluate(model, valid_loader, device)
        val_score = val_metrics.get("rmse_mean", float("inf"))

        if args.use_swanlab and _SWANLAB_AVAILABLE:
            payload = {
                "epoch": epoch,
                "train/loss": train_loss,
            }
            payload.update({f"valid/{k}": v for k, v in val_metrics.items()})
            swanlab.log(payload)

        if val_score < best_val:
            best_val = val_score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(args.output_dir, "best_model.pt"))

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    id_metrics, _, _ = evaluate(model, test_loader, device)
    save_json(Path(args.output_dir) / "metrics_id_test.json", id_metrics)

    id_predictions = collect_predictions(model, test_loader, device)
    save_json(Path(args.output_dir) / "predictions_id_test.json", id_predictions)

    if ood_loader is not None and ood_records:
        ood_metrics, _, _ = evaluate(model, ood_loader, device)
        save_json(Path(args.output_dir) / "metrics_ood_test.json", ood_metrics)

        ood_predictions = collect_predictions(model, ood_loader, device)
        save_json(Path(args.output_dir) / "predictions_ood_test.json", ood_predictions)

    if args.use_swanlab and _SWANLAB_AVAILABLE:
        payload = {f"test/{k}": v for k, v in id_metrics.items()}
        swanlab.log(payload)
        swanlab.finish()


if __name__ == "__main__":
    main()
