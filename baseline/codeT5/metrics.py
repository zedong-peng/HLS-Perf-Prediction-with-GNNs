from __future__ import annotations

from typing import Dict, Optional

import numpy as np


AVAILABLE_RESOURCES = {
    "dsp": 9024.0,
    "lut": 1_303_680.0,
    "ff": 2_607_360.0,
}


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, resource_key: Optional[str] = None) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors)))
    mse = float(np.mean(errors ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-8))
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }
    if resource_key is not None and resource_key in AVAILABLE_RESOURCES:
        metrics["ulti_rmse"] = rmse / AVAILABLE_RESOURCES[resource_key]
    return metrics
