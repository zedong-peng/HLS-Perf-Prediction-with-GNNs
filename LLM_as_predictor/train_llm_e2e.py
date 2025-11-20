#!/usr/bin/env python3
"""
QoR regression with frozen Qwen coder representations.

Workflow
========
1. Load design metadata and render deterministic textual templates.
2. Run Qwen2.5-Coder forward passes (frozen) to cache hidden-state mean pooled
   representations for several candidate layers.
3. Probe the candidates to pick the best-performing layer on the validation set.
4. Standardise features/targets and train a lightweight MLP regression head to
   predict LUT, DSP, FF, and latency_cycles simultaneously.
5. Evaluate on ID/OOD splits (after inverse transforms) and export artefacts.
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError as exc:  # pragma: no cover
    raise ImportError("transformers is required for this script") from exc

try:
    import swanlab
except ImportError as exc:  # pragma: no cover
    raise ImportError("swanlab is required for this script") from exc

try:
    from dataset_csv import embed_source_code, gather_csynth_data
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("dataset_csv.py must be importable from LLM_as_predictor") from exc

TARGET_METRICS = ("LUT", "DSP", "FF")
LOG_TARGETS = {"LUT", "FF"}
CACHE_VERSION = "coder_v20241005"
TEMPLATE_VERSION = "hls_qor_template_v1"


@dataclass
class Record:
    text: str
    labels: np.ndarray
    meta: Dict[str, Any]


class TextDataset(Dataset):
    def __init__(self, texts: Sequence[str], labels: Optional[Sequence[np.ndarray]] = None):
        self.texts = list(texts)
        self.labels = None if labels is None else np.stack(labels, axis=0)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        if self.labels is None:
            return text
        return text, self.labels[idx]


class RegressionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(TARGET_METRICS)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _extract_metadata_from_path(path_str: str) -> Dict[str, str]:
    path = Path(path_str)
    parts = path.parts
    result = {
        "source_name": parts[-8] if len(parts) >= 8 else "unknown",
        "algo_name": parts[-7] if len(parts) >= 7 else "unknown",
        "design_id": parts[-6] if len(parts) >= 6 else "unknown",
    }
    return result


def _count_pragmas(source_text: str) -> Dict[str, int]:
    pragmas = ["PIPELINE", "UNROLL", "DATAFLOW", "ARRAY_PARTITION", "INLINE"]
    counts = {}
    text_upper = source_text.upper()
    for pragma in pragmas:
        counts[pragma.lower()] = text_upper.count(f"#PRAGMA HLS {pragma}")
    counts["pragma_total"] = sum(counts.values())
    counts["loop_count"] = source_text.count("for (") + source_text.count("while (")
    counts["function_calls"] = source_text.count("(") - counts["loop_count"]
    return counts


def collect_design_dataframe(design_dir: str) -> pd.DataFrame:
    design_dir = os.path.abspath(design_dir)
    if not os.path.isdir(design_dir):
        raise FileNotFoundError(f"Design directory not found: {design_dir}")

    tmp_root = Path("./tmp_llm_qor")
    tmp_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="coder_qor_", dir=tmp_root) as tmp_dir:
        raw_csv = os.path.join(tmp_dir, "raw.csv")
        gather_csynth_data(design_dir, raw_csv)
        df = pd.read_csv(raw_csv)

    if df.empty:
        return df

    df["File Path"] = df["File Path"].astype(str)
    df = embed_source_code(df)
    metadata = df["File Path"].apply(_extract_metadata_from_path)
    df["source_name"] = metadata.apply(lambda m: m["source_name"])
    df["algo_name"] = metadata.apply(lambda m: m["algo_name"])
    df["design_id"] = metadata.apply(lambda m: m["design_id"])

    # Numeric conversions
    df["LUT"] = pd.to_numeric(df["LUT"], errors="coerce")
    df["DSP"] = pd.to_numeric(df["DSP"], errors="coerce")
    df["FF"] = pd.to_numeric(df["FF"], errors="coerce")
    df["Best-caseLatency"] = pd.to_numeric(df["Best-caseLatency"], errors="coerce")
    df["TargetClockPeriod"] = pd.to_numeric(df["TargetClockPeriod"], errors="coerce")
    df["Part"] = df["Part"].fillna("unknown")

    df = df.dropna(subset=["LUT", "DSP", "FF", "Best-caseLatency"])
    df = df.reset_index(drop=True)
    return df


def format_code_blocks(source_code: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
    code_sections = []
    joined_texts = []
    for entry in source_code:
        fname = entry.get("file_name", "")
        if not fname.endswith((".c", ".cpp", ".h", ".hpp")):
            continue
        content = entry.get("file_content", "")
        code_sections.append(f"// File: {fname}\n{content}")
        joined_texts.append(content)
    code = "\n\n".join(code_sections)
    context = {
        "num_files": len(code_sections),
        "total_characters": sum(len(text) for text in joined_texts),
    }
    if joined_texts:
        aggregate = "\n".join(joined_texts)
        context.update(_count_pragmas(aggregate))
    return code, context


def render_template(row: pd.Series) -> Record:
    source_code = row.get("source_code", [])
    code, context_counts = format_code_blocks(source_code)
    context_payload = {
        "source_name": row.get("source_name", "unknown"),
        "algo_name": row.get("algo_name", "unknown"),
        "design_id": row.get("design_id", "unknown"),
        **context_counts,
    }
    device = row.get("Part", "unknown")
    tool = "vitis_hls"
    clock_period = row.get("TargetClockPeriod", None)
    clock_mhz = None
    if isinstance(clock_period, (int, float)) and clock_period > 0:
        clock_mhz = 1000.0 / float(clock_period)
    else:
        clock_mhz = None
    context_json = json.dumps(context_payload, ensure_ascii=False, sort_keys=True)
    text = (
        "[HLS_QOR_SAMPLE]\n"
        f"device: {device}\n"
        f"tool: {tool}\n"
        f"clock_mhz: {clock_mhz if clock_mhz is not None else 'unknown'}\n"
        "task: QoR_regression\n"
        "target_metrics: [LUT, DSP, FF, latency_cycles]\n"
        f"context: {context_json}\n"
        "code:\n"
        f"{code}\n"
        "[/HLS_QOR_SAMPLE]"
    )

    labels = np.array([
        float(row["LUT"]),
        float(row["DSP"]),
        float(row["FF"]),
        float(row["Best-caseLatency"]),
    ], dtype=np.float64)

    meta = {
        "file_path": row.get("File Path", ""),
        "source_name": row.get("source_name", "unknown"),
        "algo_name": row.get("algo_name", "unknown"),
        "design_id": row.get("design_id", "unknown"),
        "clock_period": clock_period,
        "device": device,
    }
    return Record(text=text, labels=labels, meta=meta)


def build_records(df: pd.DataFrame) -> List[Record]:
    records: List[Record] = []
    for _, row in df.iterrows():
        record = render_template(row)
        records.append(record)
    return records


def collate_tokeniser(tokenizer, max_length: int):
    def _collate(batch: Sequence[str]) -> Dict[str, torch.Tensor]:
        encoded = tokenizer(
            list(batch),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return encoded

    return _collate


def cache_path(cache_dir: Path, design_dir: str, model_name: str, max_length: int, layer_fractions: Sequence[float]) -> Path:
    payload = {
        "version": CACHE_VERSION,
        "design_dir": os.path.abspath(design_dir),
        "model": model_name,
        "max_length": max_length,
        "fractions": list(layer_fractions),
        "template_version": TEMPLATE_VERSION,
    }
    key = json.dumps(payload, sort_keys=True).encode("utf-8")
    digest = hashlib.md5(key).hexdigest()
    return cache_dir / f"embeddings_{digest}.pt"


def compute_candidate_layers(num_layers: int, fractions: Sequence[float]) -> List[int]:
    candidates = set()
    for frac in fractions:
        idx = int(math.floor(num_layers * frac))
        idx = max(1, min(num_layers, idx))
        candidates.add(idx)
    candidates.add(num_layers)
    return sorted(candidates)


def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(hidden_states)
    summed = (hidden_states * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


def extract_embeddings(
    texts: Sequence[str],
    tokenizer,
    model,
    device: torch.device,
    max_length: int,
    batch_size: int,
    candidate_layers: Sequence[int],
    use_bf16: bool,
) -> Dict[int, np.ndarray]:
    dataloader = DataLoader(
        TextDataset(texts),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_tokeniser(tokenizer, max_length),
    )
    storage = {layer: [] for layer in candidate_layers}

    dtype = torch.bfloat16 if use_bf16 and torch.cuda.is_available() else torch.float32
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states
            for layer_idx in candidate_layers:
                pooled = mean_pool(hidden_states[layer_idx].to(dtype), attention_mask)
                storage[layer_idx].append(pooled.cpu().to(torch.float32).numpy())

    for layer_idx, chunks in storage.items():
        storage[layer_idx] = np.concatenate(chunks, axis=0)
    return storage


def load_or_build_embeddings(
    cache_dir: Path,
    design_dir: str,
    model_name: str,
    tokenizer,
    model,
    device: torch.device,
    max_length: int,
    batch_size: int,
    layer_fractions: Sequence[float],
    use_bf16: bool,
    rebuild_cache: bool,
    texts: Sequence[str],
    labels: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path(cache_dir, design_dir, model_name, max_length, layer_fractions)

    if cache_file.exists() and not rebuild_cache:
        payload = torch.load(cache_file)
        stored_labels = payload.get("labels")
        embeddings = {int(k): v for k, v in payload["embeddings"].items()}
        return stored_labels, embeddings

    raise_missing_labels = False
    if len(texts) == 0:
        raise ValueError("No texts provided for embedding extraction")

    candidate_layers = compute_candidate_layers(model.config.num_hidden_layers, layer_fractions)
    embeddings = extract_embeddings(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
        candidate_layers=candidate_layers,
        use_bf16=use_bf16,
    )
    stored_labels = labels if labels is not None else np.zeros((len(texts), len(TARGET_METRICS)), dtype=np.float32)
    torch.save({"embeddings": embeddings, "labels": stored_labels}, cache_file)
    return stored_labels, embeddings


def probe_best_layer(
    candidate_layers: Sequence[int],
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    embeddings: Dict[int, np.ndarray],
    labels: np.ndarray,
) -> int:
    best_layer = candidate_layers[-1]
    best_score = float("inf")

    y_train = labels[train_idx]
    y_val = labels[val_idx]

    for layer in candidate_layers:
        X_train = embeddings[layer][train_idx]
        X_val = embeddings[layer][val_idx]
        # Simple ridge regression probe
        X_t = np.concatenate([X_train, np.ones((X_train.shape[0], 1))], axis=1)
        reg = 1e-3
        XtX = X_t.T @ X_t + reg * np.eye(X_t.shape[1])
        XtY = X_t.T @ y_train
        weights = np.linalg.solve(XtX, XtY)
        preds = X_val @ weights[:-1] + weights[-1]
        mae = np.mean(np.abs(preds - y_val))
        if mae < best_score:
            best_score = mae
            best_layer = layer
    return best_layer


def apply_target_transform(y: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    transformed = y.copy()
    log_mask = [metric in LOG_TARGETS for metric in TARGET_METRICS]
    for idx, use_log in enumerate(log_mask):
        if use_log:
            transformed[:, idx] = np.log1p(transformed[:, idx])
    mean = transformed.mean(axis=0)
    std = transformed.std(axis=0)
    std[std < 1e-6] = 1.0
    transformed = (transformed - mean) / std
    stats = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "log_mask": log_mask,
        "targets": list(TARGET_METRICS),
    }
    return transformed, stats


def inverse_target_transform(pred: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
    mean = np.array(stats["mean"], dtype=np.float64)
    std = np.array(stats["std"], dtype=np.float64)
    log_mask = stats["log_mask"]
    values = pred * std + mean
    for idx, use_log in enumerate(log_mask):
        if use_log:
            values[:, idx] = np.expm1(values[:, idx])
    return values


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    per_metric_results = []
    for idx, name in enumerate(TARGET_METRICS):
        true_col = y_true[:, idx]
        pred_col = y_pred[:, idx]
        mae = mean_absolute_error(true_col, pred_col)
        rmse = math.sqrt(mean_squared_error(true_col, pred_col))
        try:
            r2 = r2_score(true_col, pred_col)
        except ValueError:
            r2 = float("nan")
        metrics[f"mae/{name}"] = mae
        metrics[f"rmse/{name}"] = rmse
        metrics[f"r2/{name}"] = r2
        per_metric_results.append((mae, rmse, r2))
    if per_metric_results:
        mae_vals, rmse_vals, r2_vals = zip(*per_metric_results)
        metrics["mae/avg"] = float(np.mean(mae_vals))
        metrics["rmse/avg"] = float(np.mean(rmse_vals))
        metrics["r2/avg"] = float(np.mean(r2_vals))
    return metrics


def prepare_feature_datasets(
    embeddings: Dict[int, np.ndarray],
    best_layer: int,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    test_idx: Sequence[int],
    ood_idx: Optional[Sequence[int]],
    labels: np.ndarray,
) -> Tuple[StandardScaler, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    X = embeddings[best_layer]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])
    X_test = scaler.transform(X[test_idx])
    X_ood = scaler.transform(X[ood_idx]) if ood_idx is not None else None

    y_train = labels[train_idx]
    y_val = labels[val_idx]
    y_test = labels[test_idx]
    y_ood = labels[ood_idx] if ood_idx is not None else None

    return scaler, X_train, X_val, X_test, X_ood, y_train, y_val, y_test, y_ood


def create_dataloader(features: np.ndarray, targets: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(features.astype(np.float32)),
        torch.from_numpy(targets.astype(np.float32)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_regressor(
    device: torch.device,
    input_dim: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    warmup_ratio: float,
    patience: int,
) -> Tuple[RegressionHead, Dict[str, float], Dict[str, Any]]:
    model = RegressionHead(in_dim=input_dim).to(device)
    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = create_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)

    total_steps = epochs * len(train_loader)
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_state = None
    best_metric = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            scheduler.step()

        val_mae = evaluate_mae(model, val_loader, device)
        if val_mae < best_metric:
            best_metric = val_mae
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics_val = evaluate_metrics(model, val_loader, device)
    return model, metrics_val, {"best_val_mae": best_metric}


def evaluate_mae(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    criterion = nn.L1Loss()
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            total += batch_x.size(0)
    if total == 0:
        return float("inf")
    return total_loss / total


def evaluate_metrics(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    preds_list = []
    targets_list = []
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            preds = model(batch_x).cpu().numpy()
            preds_list.append(preds)
            targets_list.append(batch_y.numpy())
    if not preds_list:
        return {f"mae/{name}": float("nan") for name in TARGET_METRICS}
    preds = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    return compute_metrics(targets, preds)


def evaluate_split(
    model: nn.Module,
    features: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
    y_stats: Dict[str, Any],
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    dataloader = create_dataloader(features, targets, batch_size=1024, shuffle=False)
    model.eval()
    preds_list = []
    targets_list = []
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            preds = model(batch_x).cpu().numpy()
            preds_list.append(preds)
            targets_list.append(batch_y.numpy())
    preds = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    preds_real = inverse_target_transform(preds, y_stats)
    targets_real = inverse_target_transform(targets, y_stats)
    metrics = compute_metrics(targets_real, preds_real)
    return metrics, preds_real, targets_real


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coder-based QoR regression")
    parser.add_argument("--design_base_dir", type=str, required=True)
    parser.add_argument("--ood_design_base_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="./coder_outputs")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--model_name", type=str, default="/home/user/zedongpeng/workspace/GiT/zedong/Code-Verification/Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--embed_batch_size", type=int, default=8)
    parser.add_argument("--layer_probe_fractions", type=str, default="0.6,0.7,0.8,0.9")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--disable_layer_probe", action="store_true")
    parser.add_argument("--selected_layer", type=int, default=-1)
    parser.add_argument("--disable_swanlab", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if total_ratio > 1.0 + 1e-6:
        raise ValueError("Train/val/test ratios must sum to <= 1.0")
    return args


def main(args: argparse.Namespace) -> None:
    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not args.disable_swanlab:
        swanlab.init(
            project="HLS-LLM-QoR",
            experiment_name=f"coder_qor_{random.randint(0, 99999):05d}",
            config=vars(args),
            logdir=args.output_dir,
        )

    id_df = collect_design_dataframe(args.design_base_dir)
    if id_df.empty:
        raise RuntimeError("No valid designs found in the ID dataset")
    id_records = build_records(id_df)

    texts = [rec.text for rec in id_records]
    labels = np.stack([rec.labels for rec in id_records], axis=0)

    num_samples = len(id_records)
    train_end = int(num_samples * args.train_ratio)
    val_end = int(num_samples * (args.train_ratio + args.val_ratio))
    indices = list(range(num_samples))
    random.shuffle(indices)
    train_idx = sorted(indices[:train_end])
    val_idx = sorted(indices[train_end:val_end])
    test_idx = sorted(indices[val_end:])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    if getattr(model.config, "sliding_window", None):
        model.config.sliding_window = None

    layer_fractions = [float(x) for x in args.layer_probe_fractions.split(",") if x.strip()]
    candidate_layers = compute_candidate_layers(model.config.num_hidden_layers, layer_fractions)

    cache_embeddings_dir = cache_dir / "embeddings"
    cache_embeddings_dir.mkdir(parents=True, exist_ok=True)

    _, embeddings = load_or_build_embeddings(
        cache_dir=cache_embeddings_dir,
        design_dir=args.design_base_dir,
        model_name=args.model_name,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=args.max_length,
        batch_size=args.embed_batch_size,
        layer_fractions=layer_fractions,
        use_bf16=not args.cpu,
        rebuild_cache=args.rebuild_cache,
        texts=texts,
        labels=labels,
    )

    if args.disable_layer_probe and args.selected_layer < 0:
        raise ValueError("Layer probe disabled but no selected layer provided")

    if args.disable_layer_probe:
        best_layer = args.selected_layer
    else:
        best_layer = probe_best_layer(candidate_layers, train_idx, val_idx, embeddings, labels)

    scaler, X_train, X_val, X_test, _, y_train_raw, y_val_raw, y_test_raw, _ = prepare_feature_datasets(
        embeddings=embeddings,
        best_layer=best_layer,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        ood_idx=None,
        labels=labels,
    )

    y_train, y_stats = apply_target_transform(y_train_raw)
    y_mean = np.array(y_stats["mean"])
    y_std = np.array(y_stats["std"])

    def _transform_targets(raw: np.ndarray) -> np.ndarray:
        values = raw.copy()
        for idx, use_log in enumerate(y_stats["log_mask"]):
            if use_log:
                values[:, idx] = np.log1p(values[:, idx])
        return (values - y_mean) / y_std

    y_val = _transform_targets(y_val_raw)
    y_test = _transform_targets(y_test_raw)

    model_head, val_metrics, training_info = train_regressor(
        device=device,
        input_dim=X_train.shape[1],
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        patience=args.patience,
    )

    test_metrics, preds_test, targets_test = evaluate_split(model_head, X_test, y_test, device, y_stats)

    ood_metrics = None
    preds_ood = None
    targets_ood = None
    if args.ood_design_base_dir:
        ood_df = collect_design_dataframe(args.ood_design_base_dir)
        if not ood_df.empty:
            ood_records = build_records(ood_df)
            ood_texts = [rec.text for rec in ood_records]
            ood_labels = np.stack([rec.labels for rec in ood_records], axis=0)
            _, embeddings_ood = load_or_build_embeddings(
                cache_dir=cache_embeddings_dir,
                design_dir=args.ood_design_base_dir,
                model_name=args.model_name,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_length=args.max_length,
                batch_size=args.embed_batch_size,
                layer_fractions=layer_fractions,
                use_bf16=not args.cpu,
                rebuild_cache=args.rebuild_cache,
                texts=ood_texts,
                labels=ood_labels,
            )
            X_ood = scaler.transform(embeddings_ood[best_layer])
            y_ood = _transform_targets(ood_labels)
            ood_metrics, preds_ood, targets_ood = evaluate_split(
                model_head, X_ood, y_ood, device, y_stats
            )

    artefacts = {
        "model": model_head.state_dict(),
        "x_scaler": scaler,
        "y_stats": y_stats,
        "config": {
            "model_name": args.model_name,
            "layer_index": best_layer,
            "max_length": args.max_length,
            "template_version": TEMPLATE_VERSION,
            "target_metrics": TARGET_METRICS,
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "ood_metrics": ood_metrics,
        "training_info": training_info,
    }

    torch.save(model_head.state_dict(), os.path.join(args.output_dir, "model.pt"))
    joblib.dump(scaler, os.path.join(args.output_dir, "x_scaler.pkl"))
    with open(os.path.join(args.output_dir, "y_stats.json"), "w", encoding="utf-8") as f:
        json.dump(y_stats, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(artefacts["config"], f, ensure_ascii=False, indent=2)
    metrics_payload = {
        "val": {k: float(v) for k, v in val_metrics.items()},
        "test": {k: float(v) for k, v in test_metrics.items()},
    }
    if ood_metrics is not None:
        metrics_payload["ood"] = {k: float(v) for k, v in ood_metrics.items()}
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)
    np.savez_compressed(
        os.path.join(args.output_dir, "test_predictions.npz"),
        y_true=targets_test,
        y_pred=preds_test,
    )
    if ood_metrics is not None and preds_ood is not None and targets_ood is not None:
        np.savez_compressed(
            os.path.join(args.output_dir, "ood_predictions.npz"),
            y_true=targets_ood,
            y_pred=preds_ood,
        )

    if not args.disable_swanlab:
        swanlab.log({f"val/{k}": v for k, v in val_metrics.items()})
        swanlab.log({f"test/{k}": v for k, v in test_metrics.items()})
        if ood_metrics is not None:
            swanlab.log({f"ood/{k}": v for k, v in ood_metrics.items()})
        swanlab.finish()

    print("Best layer index:", best_layer)
    print("Validation metrics:", val_metrics)
    print("Test metrics:", test_metrics)
    if ood_metrics is not None:
        print("OOD metrics:", ood_metrics)


if __name__ == "__main__":
    try:
        parsed_args = parse_args()
        main(parsed_args)
    except KeyboardInterrupt:
        print("Interrupted by user")
        if "swanlab" in sys.modules:
            swanlab.finish()
        sys.exit(1)
    except Exception as exc:
        if "swanlab" in sys.modules:
            swanlab.finish()
        raise exc
