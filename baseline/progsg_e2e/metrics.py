"""Utility functions for QoR regression metrics."""

from __future__ import annotations

from typing import Dict

import torch

from model import TARGET_NAMES

AVAILABLE_RESOURCES = {
    "dsp": 9024.0,
    "lut": 1303680.0,
    "ff": 2607360.0,
}


def compute_regression_metrics(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, Dict[str, float]]:
    """Compute MAE, RMSE, masked MAPE, ulti-RMSE (where applicable), and R² per metric."""

    if preds.ndim != 2 or targets.ndim != 2:
        raise ValueError("preds and targets must be 2D tensors [N, num_targets]")
    if preds.shape != targets.shape:
        raise ValueError("preds and targets must share the same shape")

    metrics: Dict[str, Dict[str, float]] = {}
    errors = preds - targets
    mae = torch.mean(torch.abs(errors), dim=0)
    rmse = torch.sqrt(torch.mean(errors ** 2, dim=0))

    nonzero_mask = targets != 0
    safe_targets = torch.where(nonzero_mask, targets, torch.ones_like(targets))
    abs_percent_errors = torch.zeros_like(errors)
    abs_percent_errors = torch.where(nonzero_mask, torch.abs(errors / safe_targets), abs_percent_errors)
    nonzero_count = nonzero_mask.sum(dim=0)
    safe_count = torch.clamp(nonzero_count, min=1).to(abs_percent_errors.dtype)
    mape = (abs_percent_errors.sum(dim=0) / safe_count) * 100.0
    mape = torch.where(nonzero_count > 0, mape, torch.zeros_like(mape))

    target_means = torch.mean(targets, dim=0)
    ss_res = torch.sum(errors ** 2, dim=0)
    ss_tot = torch.sum((targets - target_means) ** 2, dim=0)
    r2 = 1.0 - (ss_res / torch.clamp(ss_tot, min=1e-8))

    for idx, name in enumerate(TARGET_NAMES):
        metrics[name] = {
            "mae": float(mae[idx].item()),
            "rmse": float(rmse[idx].item()),
            "mape": float(mape[idx].item()),
            "r2": float(r2[idx].item()),
        }
        if name in AVAILABLE_RESOURCES:
            metrics[name]["ulti_rmse"] = float(rmse[idx].item() / AVAILABLE_RESOURCES[name])
    return metrics


def stack_results(accumulator: Dict[str, list], batch_preds: torch.Tensor, batch_targets: torch.Tensor) -> None:
    """Append batch predictions/targets to Python lists for later aggregation."""

    accumulator.setdefault("preds", []).append(batch_preds.detach().cpu())
    accumulator.setdefault("targets", []).append(batch_targets.detach().cpu())


def finalize_stack(accumulator: Dict[str, list]) -> Dict[str, torch.Tensor]:
    """Convert stacked batch tensors back to single tensors."""

    return {
        key: torch.cat(value, dim=0) if value else torch.empty((0, len(TARGET_NAMES)))
        for key, value in accumulator.items()
    }


__all__ = [
    "compute_regression_metrics",
    "AVAILABLE_RESOURCES",
    "stack_results",
    "finalize_stack",
]
