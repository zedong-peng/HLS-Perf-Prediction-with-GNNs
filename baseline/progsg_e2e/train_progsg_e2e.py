"""Train a ProgSG-inspired GNN on ForgeHLS design graphs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    ReduceLROnPlateau,
    StepLR,
)
from torch.utils.data import Dataset, Subset
from torch_geometric.loader import DataLoader

from dataset import (
    DesignGraphProcessor,
    DesignSample,
    assign_sample_indices,
    attach_code_features,
)
from model import ProgSGStyleModel, ProgSGMultimodalModel, TARGET_NAMES
from code_features import CodeEncoderConfig, SourceCodeFeatureCache
from metrics import (
    AVAILABLE_RESOURCES,
    compute_regression_metrics,
    finalize_stack,
    stack_results,
)

LOSS_TARGET_INDICES = tuple(idx for idx, name in enumerate(TARGET_NAMES) if name != "latency")


def _is_metric_payload_valid(metrics: Dict[str, float]) -> bool:
    """Ensure resource metrics do not exceed device capacities."""

    if not metrics:
        return False
    for raw_key, raw_value in metrics.items():
        normalized_key = raw_key.lower()
        if normalized_key not in AVAILABLE_RESOURCES:
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if value > AVAILABLE_RESOURCES[normalized_key]:
            return False
    return True


def _is_sample_payload_valid(sample: DesignSample) -> bool:
    """Mirror delta_e2e cache validation on design-only samples."""

    if not _is_metric_payload_valid(sample.metrics):
        return False
    graph_targets = getattr(sample.graph, "y", None)
    if graph_targets is None:
        return False
    if not torch.isfinite(graph_targets).all():
        return False
    return True


def _filter_invalid_samples(samples: List[DesignSample]) -> List[DesignSample]:
    """Drop cached samples whose metrics exceed platform resources."""

    valid_samples: List[DesignSample] = []
    for sample in samples:
        if _is_sample_payload_valid(sample):
            valid_samples.append(sample)
    filtered = len(samples) - len(valid_samples)
    if filtered:
        print(f"Filtered {filtered} cached design samples that violated resource limits.")
    return valid_samples


def report_numerical_issue(
    *,
    stage: str,
    epoch: int,
    step: int,
    loss_value: torch.Tensor,
    preds: torch.Tensor,
    preds_norm: torch.Tensor,
    targets_norm: torch.Tensor,
    batch,
    dataset: Optional["DesignGraphDataset"],
) -> None:
    """Print diagnostics when non-finite values appear during training/eval."""

    def describe(name: str, tensor: torch.Tensor) -> str:
        flat = tensor.detach().flatten().cpu()
        finite = torch.isfinite(flat)
        return (
            f"{name}(shape={tuple(tensor.shape)}, finite={finite.all().item()}, "
            f"min={flat.min().item():.6g}, max={flat.max().item():.6g})"
        )

    print(f"[{stage}] Non-finite values at epoch {epoch}, step {step}")
    print(f"  loss={loss_value.item()} finite={torch.isfinite(loss_value).item()}")
    print("  " + describe("preds", preds))
    print("  " + describe("preds_norm", preds_norm))
    print("  " + describe("targets_norm", targets_norm))

    if dataset is not None and hasattr(batch, "sample_id"):
        sample_ids = batch.sample_id.detach().cpu().view(-1).tolist()
        unique_ids = sorted({int(idx) for idx in sample_ids})
        print(f"  affected_samples={len(unique_ids)}")
        for sid in unique_ids:
            meta = dataset.metadata(sid)
            meta_str = ", ".join(f"{k}={v}" for k, v in meta.items())
            print(f"    sample_id={sid}: {meta_str}")
    else:
        print("  sample metadata unavailable for this batch.")


class DesignGraphDataset(Dataset):
    """Torch dataset that wraps parsed design samples."""

    def __init__(self, samples):
        self.samples = samples
        self.code_template = self._build_code_template()

    def __len__(self) -> int:
        return len(self.samples)

    def _build_code_template(self) -> Dict[str, torch.Tensor]:
        for sample in self.samples:
            if sample.code_inputs:
                template: Dict[str, torch.Tensor] = {}
                for key, value in sample.code_inputs.items():
                    tensor = value.detach().cpu().clone().contiguous()
                    if key == "pooled_embedding" and tensor.dim() == 1:
                        tensor = tensor.unsqueeze(0)
                    template[key] = tensor
                return template
        return {}

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        graph = sample.graph.clone()
        if sample.code_inputs:
            seq_len = None
            for key, value in sample.code_inputs.items():
                attr_name = f"code_{key}"
                tensor = value.detach().cpu().clone().contiguous()
                if key == "pooled_embedding" and tensor.dim() == 1:
                    tensor = tensor.unsqueeze(0)
                graph.__setattr__(attr_name, tensor)
                if key == "attention_mask":
                    seq_len = int(value.shape[0])
                elif seq_len is None and isinstance(value, torch.Tensor):
                    seq_len = int(value.shape[0])
            if seq_len is None:
                seq_len = 0
            graph.code_seq_len = torch.tensor([seq_len], dtype=torch.long)
            graph.has_code = torch.tensor([1], dtype=torch.long)
            for key, template in self.code_template.items():
                attr_name = f"code_{key}"
                if not hasattr(graph, attr_name):
                    zero_tensor = torch.zeros_like(template)
                    graph.__setattr__(attr_name, zero_tensor)
        else:
            graph.has_code = torch.tensor([0], dtype=torch.long)
            graph.code_seq_len = torch.tensor([0], dtype=torch.long)
            for key, template in self.code_template.items():
                attr_name = f"code_{key}"
                zero_tensor = torch.zeros_like(template)
                graph.__setattr__(attr_name, zero_tensor)
        return graph

    def metadata(self, idx: int) -> Dict[str, str]:
        sample = self.samples[idx]
        return {
            "source_name": sample.source_name,
            "algo_name": sample.algo_name,
            "design_id": sample.design_id,
        }


def compute_label_stats(dataset: DesignGraphDataset, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    labels: List[torch.Tensor] = []
    for idx in indices.tolist():
        y = dataset.samples[idx].graph.y.squeeze(0)
        labels.append(y)
    y_stack = torch.stack(labels, dim=0)
    mean = y_stack.mean(dim=0)
    std = y_stack.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return mean, std


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_indices(total_size: int, train_ratio: float, valid_ratio: float, seed: int) -> Dict[str, torch.Tensor]:
    if total_size == 0:
        return {"train": torch.empty(0, dtype=torch.long), "valid": torch.empty(0, dtype=torch.long), "test": torch.empty(0, dtype=torch.long)}

    gen = torch.Generator()
    gen.manual_seed(seed)
    perm = torch.randperm(total_size, generator=gen)
    train_size = int(total_size * train_ratio)
    valid_size = int(total_size * valid_ratio)
    test_size = total_size - train_size - valid_size
    if train_size == 0 or valid_size == 0 or test_size == 0:
        raise ValueError("Dataset too small for the requested split ratios")
    return {
        "train": perm[:train_size],
        "valid": perm[train_size:train_size + valid_size],
        "test": perm[train_size + valid_size:],
    }


def data_loader_from_subset(dataset: torch.utils.data.Dataset, indices: torch.Tensor, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    subset = Subset(dataset, indices.tolist())
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def evaluate(
    model: ProgSGStyleModel,
    loader: DataLoader,
    device: torch.device,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    loss_indices: Optional[Tuple[int, ...]] = None,
    *,
    debug_nan: bool = False,
    dataset: Optional[DesignGraphDataset] = None,
) -> Dict[str, object]:
    model.eval()
    loss_fn = nn.MSELoss(reduction="mean")
    total_loss = 0.0
    total_samples = 0
    accumulator: Dict[str, List[torch.Tensor]] = {}
    indices: List[torch.Tensor] = []
    mean = target_mean.to(device)
    std = target_std.to(device)
    with torch.no_grad():
        for step_idx, batch in enumerate(loader, start=1):
            batch = batch.to(device)
            preds = model(batch)
            targets = batch.y.squeeze(1)
            preds_norm = (preds - mean) / std
            targets_norm = (targets - mean) / std
            if loss_indices:
                loss = loss_fn(preds_norm[:, loss_indices], targets_norm[:, loss_indices])
            else:
                loss = loss_fn(preds_norm, targets_norm)
            if debug_nan:
                tensors_ok = (
                    torch.isfinite(loss)
                    and torch.isfinite(preds).all()
                    and torch.isfinite(preds_norm).all()
                    and torch.isfinite(targets_norm).all()
                )
                if not tensors_ok:
                    nonfinite_params = [
                        name
                        for name, param in model.named_parameters()
                        if not torch.isfinite(param).all()
                    ]
                    if nonfinite_params:
                        print("  Detected non-finite parameters during eval:", ", ".join(nonfinite_params))
                    report_numerical_issue(
                        stage="eval",
                        epoch=-1,
                        step=step_idx,
                        loss_value=loss,
                        preds=preds,
                        preds_norm=preds_norm,
                        targets_norm=targets_norm,
                        batch=batch,
                        dataset=dataset,
                    )
                    raise RuntimeError("Non-finite values detected during evaluation.")
            total_loss += loss.item() * batch.num_graphs
            total_samples += batch.num_graphs
            stack_results(accumulator, preds, targets)
            if hasattr(batch, "sample_id"):
                indices.append(batch.sample_id.view(-1).cpu())
    if total_samples == 0:
        return {
            "loss": 0.0,
            "metrics": {},
            "preds": torch.empty(0, len(TARGET_NAMES)),
            "targets": torch.empty(0, len(TARGET_NAMES)),
            "indices": torch.empty(0, dtype=torch.long),
        }
    stacked = finalize_stack(accumulator)
    metrics = compute_regression_metrics(stacked["preds"], stacked["targets"])
    prediction_indices = torch.cat(indices, dim=0) if indices else torch.empty(0, dtype=torch.long)
    return {
        "loss": total_loss / total_samples,
        "metrics": metrics,
        "preds": stacked["preds"],
        "targets": stacked["targets"],
        "indices": prediction_indices,
    }


def save_predictions(output_dir: Path, tag: str, eval_output: Dict[str, object], dataset: DesignGraphDataset) -> None:
    records: List[Dict[str, object]] = []
    preds = eval_output["preds"].numpy()
    targets = eval_output["targets"].numpy()
    indices_tensor: torch.Tensor = eval_output["indices"]
    if indices_tensor.numel() == 0:
        return
    indices = indices_tensor.tolist()
    for row_idx, dataset_idx in enumerate(indices):
        meta = dataset.metadata(dataset_idx)
        record = {
            **meta,
            "pred": {name: float(preds[row_idx, col]) for col, name in enumerate(TARGET_NAMES)},
            "target": {name: float(targets[row_idx, col]) for col, name in enumerate(TARGET_NAMES)},
        }
        records.append(record)
    out_path = output_dir / f"predictions_{tag}.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ProgSG-style GNN on ForgeHLS")
    parser.add_argument("--design_base_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("./progsg_output"))
    parser.add_argument("--cache_root", type=Path, default=Path("./progsg_cache"))
    parser.add_argument("--ood_design_base_dir", type=str, default=None)
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--hierarchical", action="store_true")
    parser.add_argument("--region", action="store_true")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=2.0)

    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--max_designs", type=int, default=None,
                        help="Limit number of designs for faster experiments")

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_swanlab", action="store_true", help="Enable SwanLab logging when available.")
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="none",
        choices=["none", "cosine", "step", "plateau"],
        help="Optional learning rate scheduler.",
    )
    parser.add_argument("--lr_min", type=float, default=1e-6, help="Minimum learning rate for schedulers.")
    parser.add_argument(
        "--lr_step_size",
        type=int,
        default=50,
        help="Step size in epochs for StepLR when lr_scheduler=step.",
    )
    parser.add_argument(
        "--lr_gamma",
        type=float,
        default=0.5,
        help="Multiplicative decay factor for StepLR when lr_scheduler=step.",
    )
    parser.add_argument(
        "--lr_plateau_patience",
        type=int,
        default=10,
        help="Number of epochs with no improvement before ReduceLROnPlateau adjusts LR.",
    )
    parser.add_argument(
        "--lr_plateau_factor",
        type=float,
        default=0.5,
        help="Multiplicative factor for ReduceLROnPlateau when lr_scheduler=plateau.",
    )
    parser.add_argument(
        "--lr_warmup_epochs",
        type=int,
        default=0,
        help="Linear warmup epochs before applying the main scheduler.",
    )
    parser.add_argument(
        "--lr_warmup_start_factor",
        type=float,
        default=0.1,
        help="Initial LR scale during warmup (1.0 keeps the original LR).",
    )
    parser.add_argument(
        "--eval_checkpoint",
        type=str,
        default="last",
        choices=["last", "best"],
        help="Choose which checkpoint to use for ID/OOD evaluation summaries.",
    )

    parser.add_argument("--enable_code", dest="enable_code", action="store_true")
    parser.add_argument("--disable_code", dest="enable_code", action="store_false")
    parser.set_defaults(enable_code=True)

    parser.add_argument("--code_encoder_name", type=str, default="Salesforce/codet5-small")
    parser.add_argument("--code_max_length", type=int, default=256)
    parser.add_argument("--code_cache_namespace", type=str, default="codet5_small_v1")
    parser.add_argument("--code_local_files_only", action="store_true")
    parser.add_argument(
        "--code_cache_device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--code_transformer_layers", type=int, default=2)
    parser.add_argument("--code_transformer_heads", type=int, default=4)
    parser.add_argument(
        "--code_fusion_mode",
        type=str,
        default="concat",
        choices=["concat", "add"],
    )
    parser.add_argument(
        "--code_node_interaction",
        action="store_true",
        help="Enable node–token interaction attention before graph pooling",
    )
    parser.add_argument(
        "--debug_nan",
        action="store_true",
        help="Print diagnostics and abort if non-finite losses are encountered.",
    )

    args = parser.parse_args()
    use_swanlab = args.use_swanlab and _SWANLAB_AVAILABLE
    if args.use_swanlab and not _SWANLAB_AVAILABLE:
        print("SwanLab not installed; proceeding without logging.")

    if args.ood_design_base_dir:
        ood_path = Path(args.ood_design_base_dir)
        if not ood_path.exists():
            print(f"Warning: OOD design directory {ood_path} does not exist; skipping OOD evaluation.")
            args.ood_design_base_dir = None
        else:
            args.ood_design_base_dir = ood_path
    else:
        args.ood_design_base_dir = None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = args.output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoints_dir / "model_best.pt"
    last_model_path = checkpoints_dir / "model_last.pt"

    if args.enable_code and args.code_cache_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available for code caching; falling back to CPU.")
        args.code_cache_device = "cpu"

    seed_everything(args.seed)

    serialized_args = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }

    if use_swanlab:
        run_name = args.output_dir.name or "progsg_e2e"
        try:
            swanlab.init(
                project="ProgSG-E2E",
                experiment_name=run_name,
                config=serialized_args,
                logdir=str(args.output_dir),
            )
        except Exception as exc:  # pragma: no cover - network/runtime failures
            print(f"SwanLab init failed ({exc}); continuing without logging.")
            use_swanlab = False

    processor = DesignGraphProcessor(
        design_base_dir=args.design_base_dir,
        output_dir=args.output_dir,
        cache_root=args.cache_root,
        rebuild_cache=args.rebuild_cache,
        hierarchical=args.hierarchical,
        region=args.region,
        include_code=args.enable_code,
    )
    samples = processor.collect_designs(max_designs=args.max_designs, seed=args.seed)
    if not samples:
        raise RuntimeError("No design samples were parsed. Check dataset path.")
    samples = _filter_invalid_samples(samples)
    if not samples:
        raise RuntimeError("No valid design samples remained after cache filtering.")

    code_cache = None
    code_embedding_dim: Optional[int] = None
    if args.enable_code:
        code_cache_dir = args.cache_root / "code_features"
        code_config = CodeEncoderConfig(
            model_name_or_path=args.code_encoder_name,
            max_length=args.code_max_length,
            cache_namespace=args.code_cache_namespace,
            local_files_only=args.code_local_files_only,
            store_token_embeddings=True,
            store_pooled=True,
        )
        code_cache = SourceCodeFeatureCache(
            cache_root=code_cache_dir,
            config=code_config,
            device=args.code_cache_device,
        )
        attach_code_features(samples, code_cache, encode=True)

        example_inputs = next(
            (s.code_inputs for s in samples if s.code_inputs is not None),
            None,
        )
        if example_inputs is None:
            raise RuntimeError("Code modality enabled but no code features were generated.")
        if "token_embeddings" in example_inputs:
            code_embedding_dim = int(example_inputs["token_embeddings"].shape[1])
        elif "pooled_embedding" in example_inputs:
            code_embedding_dim = int(example_inputs["pooled_embedding"].shape[-1])
        else:
            raise RuntimeError("Unable to infer code embedding dimension from cached features.")

    assign_sample_indices(samples)
    dataset = DesignGraphDataset(samples)

    indices = split_indices(len(dataset), args.train_ratio, args.valid_ratio, args.seed)
    if use_swanlab:
        swanlab.log({
            "dataset/train_samples": int(indices["train"].numel()),
            "dataset/valid_samples": int(indices["valid"].numel()),
            "dataset/test_samples": int(indices["test"].numel()),
        })
    target_mean, target_std = compute_label_stats(dataset, indices["train"])
    target_mean_cpu = target_mean.clone().detach().cpu()
    target_std_cpu = target_std.clone().detach().cpu()
    train_loader = data_loader_from_subset(dataset, indices["train"], args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = data_loader_from_subset(dataset, indices["valid"], args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = data_loader_from_subset(dataset, indices["test"], args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device(args.device)
    if args.enable_code:
        if code_embedding_dim is None:
            raise RuntimeError("Code embedding dimension not initialised.")
        model = ProgSGMultimodalModel(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            heads=args.heads,
            dropout=args.dropout,
            code_embedding_dim=code_embedding_dim,
            code_transformer_layers=args.code_transformer_layers,
            code_transformer_heads=args.code_transformer_heads,
            fusion_mode=args.code_fusion_mode,
            node_token_interaction=bool(args.code_node_interaction),
        ).to(device)
    else:
        model = ProgSGStyleModel(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            heads=args.heads,
            dropout=args.dropout,
        ).to(device)

    target_mean_tensor = target_mean.to(device)
    target_std_tensor = target_std.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss(reduction="mean")
    warmup_scheduler: Optional[LinearLR] = None
    main_scheduler = None
    scheduler_requires_metric = False

    if args.lr_warmup_epochs > 0:
        warmup_iters = max(args.lr_warmup_epochs, 1)
        start_factor = max(min(args.lr_warmup_start_factor, 1.0), 1e-6)
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=start_factor,
            total_iters=warmup_iters,
        )

    scheduler_name = args.lr_scheduler.lower()
    if scheduler_name == "cosine":
        t_max = max(1, args.epochs - args.lr_warmup_epochs)
        main_scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=args.lr_min)
    elif scheduler_name == "step":
        main_scheduler = StepLR(
            optimizer,
            step_size=max(args.lr_step_size, 1),
            gamma=max(args.lr_gamma, 1e-6),
        )
    elif scheduler_name == "plateau":
        main_scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=max(min(args.lr_plateau_factor, 1.0), 1e-6),
            patience=max(args.lr_plateau_patience, 1),
            min_lr=args.lr_min,
        )
        scheduler_requires_metric = True

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_loss = float("inf")
    best_epoch: Optional[int] = None
    last_val_loss: Optional[float] = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for step_idx, batch in enumerate(train_loader, start=1):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            preds = model(batch)
            targets = batch.y.squeeze(1)
            preds_norm = (preds - target_mean_tensor) / target_std_tensor
            targets_norm = (targets - target_mean_tensor) / target_std_tensor
            if LOSS_TARGET_INDICES:
                loss = loss_fn(preds_norm[:, LOSS_TARGET_INDICES], targets_norm[:, LOSS_TARGET_INDICES])
            else:
                loss = loss_fn(preds_norm, targets_norm)
            if args.debug_nan:
                tensors_ok = (
                    torch.isfinite(loss)
                    and torch.isfinite(preds).all()
                    and torch.isfinite(preds_norm).all()
                    and torch.isfinite(targets_norm).all()
                )
                if not tensors_ok:
                    nonfinite_params = [
                        name
                        for name, param in model.named_parameters()
                        if not torch.isfinite(param).all()
                    ]
                    nonfinite_grads = [
                        name
                        for name, param in model.named_parameters()
                        if param.grad is not None and not torch.isfinite(param.grad).all()
                    ]
                    if nonfinite_params:
                        print("  Detected non-finite parameters:", ", ".join(nonfinite_params))
                    if nonfinite_grads:
                        print("  Detected non-finite gradients:", ", ".join(nonfinite_grads))
                    if args.debug_nan:
                        debug_dir = args.output_dir / "debug_nan"
                        debug_dir.mkdir(parents=True, exist_ok=True)
                        state_path = debug_dir / f"epoch{epoch:03d}_step{step_idx:04d}_state.pth"
                        torch.save(model.state_dict(), state_path)
                        sample_ids = (
                            batch.sample_id.detach().cpu().view(-1).tolist()
                            if hasattr(batch, "sample_id")
                            else []
                        )
                        payload = {
                            "epoch": epoch,
                            "step": step_idx,
                            "loss": loss.detach().cpu(),
                            "sample_ids": sample_ids,
                        }
                        torch.save(payload, debug_dir / f"epoch{epoch:03d}_step{step_idx:04d}_batch.pt")
                    report_numerical_issue(
                        stage="train",
                        epoch=epoch,
                        step=step_idx,
                        loss_value=loss,
                        preds=preds,
                        preds_norm=preds_norm,
                        targets_norm=targets_norm,
                        batch=batch,
                        dataset=dataset,
                    )
                    raise RuntimeError("Non-finite values detected during training.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            total_samples += batch.num_graphs
        train_loss = total_loss / max(total_samples, 1)

        val_output = evaluate(
            model,
            valid_loader,
            device,
            target_mean,
            target_std,
            loss_indices=LOSS_TARGET_INDICES,
            debug_nan=args.debug_nan,
            dataset=dataset,
        )
        val_loss = val_output["loss"]
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d} | LR {current_lr:.6e} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f}")
        if use_swanlab:
            payload = {
                "epoch": epoch,
                "train/loss": train_loss,
                "valid/loss": val_loss,
                "optimizer/lr": current_lr,
            }
            for target_name, metric_dict in val_output["metrics"].items():
                for metric_name, metric_value in metric_dict.items():
                    payload[f"valid/{target_name}/{metric_name}"] = float(metric_value)
            swanlab.log(payload)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            print(f"  New best model at epoch {epoch}")
            if use_swanlab:
                swanlab.log({"best/epoch": epoch, "best/val_loss": best_val_loss, "best/lr": current_lr})
            best_payload = {
                "model_state": best_state,
                "model_type": "multimodal" if args.enable_code else "graph",
                "target_names": list(TARGET_NAMES),
                "target_mean": target_mean_cpu.clone(),
                "target_std": target_std_cpu.clone(),
                "epoch": epoch,
                "val_loss": float(best_val_loss),
                "args": serialized_args,
            }
            torch.save(best_payload, best_model_path)

        if warmup_scheduler is not None and epoch <= args.lr_warmup_epochs:
            warmup_scheduler.step()
        elif main_scheduler is not None:
            if scheduler_requires_metric:
                main_scheduler.step(val_loss)
            else:
                main_scheduler.step()
        last_val_loss = float(val_loss)

    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    final_payload = {
        "model_state": final_state,
        "model_type": "multimodal" if args.enable_code else "graph",
        "target_names": list(TARGET_NAMES),
        "target_mean": target_mean_cpu.clone(),
        "target_std": target_std_cpu.clone(),
        "epoch": args.epochs,
        "val_loss": None if last_val_loss is None else float(last_val_loss),
        "best_epoch": best_epoch,
        "best_val_loss": None if best_val_loss == float("inf") else float(best_val_loss),
        "args": serialized_args,
    }
    torch.save(final_payload, last_model_path)
    print(f"\nSaved final model checkpoint to {last_model_path}")
    if best_epoch is not None:
        print(f"Current best checkpoint stored at {best_model_path} (epoch {best_epoch})")

    eval_state = final_state
    eval_label = "last"
    if args.eval_checkpoint == "best":
        if best_state is not None:
            eval_state = best_state
            eval_label = "best"
        else:
            print("Warning: best checkpoint requested for evaluation but unavailable; using last checkpoint instead.")
    model.load_state_dict(eval_state)
    print(f"Loaded {eval_label} checkpoint for evaluation.")

    id_test_output = evaluate(
        model,
        test_loader,
        device,
        target_mean,
        target_std,
        loss_indices=LOSS_TARGET_INDICES,
        debug_nan=args.debug_nan,
        dataset=dataset,
    )
    print("\nID Test Metrics:")
    for name, metric_dict in id_test_output["metrics"].items():
        print(f"  {name.upper()}: " + ", ".join([f"{k}={v:.6f}" for k, v in metric_dict.items()]))

    save_predictions(args.output_dir, "id_test", id_test_output, dataset)

    ood_output = None
    if args.ood_design_base_dir is not None and args.ood_design_base_dir.exists():
        ood_processor = DesignGraphProcessor(
            design_base_dir=args.ood_design_base_dir,
            output_dir=args.output_dir,
            cache_root=args.cache_root / "ood",
            rebuild_cache=args.rebuild_cache,
            hierarchical=args.hierarchical,
            region=args.region,
            include_code=args.enable_code,
        )
        ood_samples = ood_processor.collect_designs()
        if ood_samples:
            if args.enable_code and code_cache is not None:
                attach_code_features(ood_samples, code_cache, encode=True)
            assign_sample_indices(ood_samples)
            ood_dataset = DesignGraphDataset(ood_samples)
            ood_loader = DataLoader(
                ood_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
            ood_output = evaluate(
                model,
                ood_loader,
                device,
                target_mean,
                target_std,
                loss_indices=LOSS_TARGET_INDICES,
                debug_nan=args.debug_nan,
                dataset=ood_dataset,
            )
            print("\nOOD Test Metrics:")
            for name, metric_dict in ood_output["metrics"].items():
                print(f"  {name.upper()}: " + ", ".join([f"{k}={v:.6f}" for k, v in metric_dict.items()]))
            save_predictions(args.output_dir, "ood_test", ood_output, ood_dataset)

    if use_swanlab:
        final_payload = {
            "best/val_loss": best_val_loss,
            "id_test/loss": id_test_output["loss"],
        }
        for target_name, metric_dict in id_test_output["metrics"].items():
            for metric_name, metric_value in metric_dict.items():
                final_payload[f"id_test/{target_name}/{metric_name}"] = float(metric_value)
        if ood_output is not None:
            final_payload["ood_test/loss"] = ood_output["loss"]
            for target_name, metric_dict in ood_output["metrics"].items():
                for metric_name, metric_value in metric_dict.items():
                    final_payload[f"ood_test/{target_name}/{metric_name}"] = float(metric_value)
        final_payload["optimizer/final_lr"] = optimizer.param_groups[0]["lr"]
        swanlab.log(final_payload)

    # Persist metrics summary
    summary_path = args.output_dir / "metrics_summary.json"
    summary_payload = {
        "id_test": id_test_output["metrics"],
        "ood_test": ood_output["metrics"] if ood_output is not None else None,
        "best_val_loss": best_val_loss,
        "num_samples": {
            "train": int(indices["train"].numel()),
            "valid": int(indices["valid"].numel()),
            "test": int(indices["test"].numel()),
        },
        "hyperparams": serialized_args,
    }
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, indent=2)

    if use_swanlab:
        swanlab.finish()


if __name__ == "__main__":
    main()
