from __future__ import annotations

import json
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

try:
    import swanlab  # type: ignore

    _SWAN_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dep

    class _SwanStub:
        def init(self, *_, **__):
            return None

        def log(self, *_, **__):
            return None

        def finish(self, *_, **__):
            return None

    swanlab = _SwanStub()  # type: ignore
    _SWAN_AVAILABLE = False

from .config import CodeT5Config
from .data import CodeT5Dataset, create_dataloaders, TARGET_KEYS
from .metrics import AVAILABLE_RESOURCES, regression_metrics
from .model import CodeRegressor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_target(batch: Dict[str, torch.Tensor], metric_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    pooled = batch["pooled"].to(torch.float32)
    target = batch["target"][:, metric_idx].to(torch.float32)
    return pooled, target


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Adam,
    criterion: nn.Module,
    device: torch.device,
    metric_idx: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        optimizer.zero_grad()
        pooled, target = _select_target(batch, metric_idx)
        pooled = pooled.to(device)
        target = target.to(device)
        pred_all = model(pooled)
        pred = pred_all[:, metric_idx]
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * pooled.size(0)
        total_samples += pooled.size(0)
    return total_loss / max(total_samples, 1)


def evaluate(
    model: nn.Module,
    loader: Optional[DataLoader],
    criterion: nn.Module,
    device: torch.device,
    metric_idx: int,
    resource_key: Optional[str],
) -> Tuple[Dict[str, float], Optional[Dict[str, np.ndarray]]]:
    if loader is None or len(loader.dataset) == 0:  # type: ignore[arg-type]
        return {}, None
    model.eval()
    total_loss = 0.0
    total_samples = 0
    preds: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            pooled, target = _select_target(batch, metric_idx)
            pooled = pooled.to(device)
            target = target.to(device)
            pred_all = model(pooled)
            pred = pred_all[:, metric_idx]
            loss = criterion(pred, target)
            total_loss += float(loss.item()) * pooled.size(0)
            total_samples += pooled.size(0)
            preds.append(pred.detach().cpu())
            targets.append(target.detach().cpu())
    preds_t = torch.cat(preds).numpy()
    targets_t = torch.cat(targets).numpy()
    metrics = regression_metrics(targets_t, preds_t, resource_key=resource_key)
    metrics["mse"] = float(total_loss / max(total_samples, 1))
    metrics["rmse"] = float(np.sqrt(metrics["mse"]))
    details = {
        "y_pred": preds_t.astype(np.float64),
        "y_true": targets_t.astype(np.float64),
    }
    return metrics, details


def save_predictions(path: Path, details: Dict[str, np.ndarray]) -> None:
    df = pd.DataFrame({
        "prediction": details["y_pred"],
        "target": details["y_true"],
    })
    df.to_csv(path, index=False)


def run_training(config: CodeT5Config) -> Dict[str, object]:
    set_seed(config.seed)
    dataset, split_idx, loaders, ood_dataset, ood_loader = create_dataloaders(config)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    sample_dim = dataset[0]["pooled"].shape[-1]
    model = CodeRegressor(input_dim=sample_dim, hidden_dim=config.mlp_hidden, dropout=config.dropout).to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.MSELoss()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = config.artifact_dir(run_id)

    use_swan = _SWAN_AVAILABLE and not config.no_swanlab
    swan_run = None
    if use_swan:
        swan_config = asdict(config)
        swan_config["artifact_dir"] = str(artifact_dir)
        swan_config["run_id"] = run_id
        swan_run = swanlab.init(
            project=config.swan_project,
            experiment_name=f"{config.swan_prefix}_{config.metric_name}_{run_id}",
            config=swan_config,
            logdir=str(artifact_dir),
        )
        swanlab.log({
            "dataset/train_size": len(loaders["train"].dataset),  # type: ignore[index]
            "dataset/valid_size": len(loaders["valid"].dataset) if loaders["valid"] is not None else 0,
            "dataset/test_size": len(loaders["test"].dataset) if loaders["test"] is not None else 0,
            "dataset/ood_size": len(ood_dataset) if ood_dataset is not None else 0,
        })

    metric_idx = TARGET_KEYS.index(config.metric_name)
    resource_key = config.metric_name if config.metric_name in AVAILABLE_RESOURCES else None

    best_state = model.state_dict()
    best_val = float("inf")
    best_epoch = -1
    epochs_without_improve = 0
    history: List[Dict[str, float]] = []

    for epoch in range(config.epochs):
        train_mse = train_one_epoch(model, loaders["train"], optimizer, criterion, device, metric_idx)
        val_metrics, _ = evaluate(model, loaders["valid"], criterion, device, metric_idx, resource_key)
        val_rmse = val_metrics.get("rmse", float(np.sqrt(train_mse)))

        history.append({
            "epoch": epoch + 1,
            "train_mse": train_mse,
            "train_rmse": float(np.sqrt(train_mse)),
            "val_rmse": val_rmse,
            "val_mae": val_metrics.get("mae", float("nan")),
            "val_r2": val_metrics.get("r2", float("nan")),
            "val_ulti_rmse": val_metrics.get("ulti_rmse", float("nan")),
        })

        if use_swan:
            swan_payload = {
                "epoch": epoch + 1,
                "train/mse": train_mse,
                "train/rmse": float(np.sqrt(train_mse)),
                "valid/rmse": val_rmse,
                "valid/mae": val_metrics.get("mae"),
                "valid/r2": val_metrics.get("r2"),
                "valid/ulti_rmse": val_metrics.get("ulti_rmse"),
            }
            swanlab.log(swan_payload)

        improved = val_rmse + config.min_delta < best_val
        if improved:
            best_val = val_rmse
            best_state = model.state_dict()
            best_epoch = epoch + 1
            epochs_without_improve = 0
            if use_swan:
                swanlab.log({"best/epoch": best_epoch, "best/val_rmse": best_val})
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= config.patience:
                break

    model.load_state_dict(best_state)

    train_metrics, train_details = evaluate(model, loaders["train"], criterion, device, metric_idx, resource_key)
    val_metrics, val_details = evaluate(model, loaders["valid"], criterion, device, metric_idx, resource_key)
    test_metrics, test_details = evaluate(model, loaders["test"], criterion, device, metric_idx, resource_key)
    ood_metrics, ood_details = evaluate(model, ood_loader, criterion, device, metric_idx, resource_key)

    torch.save(best_state, artifact_dir / "best_model.pt")

    for name, details in [
        ("train", train_details),
        ("valid", val_details),
        ("test", test_details),
        ("ood_test", ood_details),
    ]:
        if details is not None:
            save_predictions(artifact_dir / f"{name}_predictions.csv", details)

    with open(artifact_dir / "history.json", "w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)

    config_dict = asdict(config)
    config_dict["artifact_dir"] = str(artifact_dir)
    config_dict["run_id"] = run_id
    with open(artifact_dir / "config.json", "w", encoding="utf-8") as fp:
        json.dump(config_dict, fp, indent=2)

    summary = {
        "metric": config.metric_name,
        "run_id": run_id,
        "artifact_dir": str(artifact_dir),
        "best_epoch": best_epoch,
        "train_metrics": train_metrics,
        "valid_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    if ood_metrics:
        summary["ood_test_metrics"] = ood_metrics

    with open(artifact_dir / "metrics.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    if use_swan:
        swan_payload = {
            "final/train_rmse": train_metrics.get("rmse"),
            "final/train_mae": train_metrics.get("mae"),
            "final/train_r2": train_metrics.get("r2"),
            "final/train_ulti_rmse": train_metrics.get("ulti_rmse"),
            "final/valid_rmse": val_metrics.get("rmse"),
            "final/valid_mae": val_metrics.get("mae"),
            "final/valid_r2": val_metrics.get("r2"),
            "final/test_rmse": test_metrics.get("rmse"),
            "final/test_mae": test_metrics.get("mae"),
            "final/test_r2": test_metrics.get("r2"),
        }
        if ood_metrics:
            swan_payload.update({
                "final/ood_rmse": ood_metrics.get("rmse"),
                "final/ood_mae": ood_metrics.get("mae"),
                "final/ood_r2": ood_metrics.get("r2"),
            })
        swanlab.log(swan_payload)
        swanlab.finish()

    print(f"[{config.metric_name}] ID test mae={test_metrics.get('mae', float('nan')):.4f} rmse={test_metrics.get('rmse', float('nan')):.4f} r2={test_metrics.get('r2', float('nan')):.4f}")
    if ood_metrics:
        print(f"[{config.metric_name}] OOD test mae={ood_metrics.get('mae', float('nan')):.4f} rmse={ood_metrics.get('rmse', float('nan')):.4f} r2={ood_metrics.get('r2', float('nan')):.4f}")

    return summary
