#!/usr/bin/env python3
"""
8:1:1 split error analysis with structured outputs (JSON + log).
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import DataLoader

try:
    from delta_e2e.train_e2e import (  # type: ignore
        E2EDifferentialDataset,
        E2EDifferentialProcessor,
        SimpleDifferentialGNN,
        METRIC_INDEX,
        canonical_metric_name,
        differential_collate_fn,
        _compute_basic_stats,
        _extract_metric_values,
    )
except Exception:
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_e2e import (
        E2EDifferentialDataset,
        E2EDifferentialProcessor,
        SimpleDifferentialGNN,
        METRIC_INDEX,
        canonical_metric_name,
        differential_collate_fn,
        _compute_basic_stats,
        _extract_metric_values,
    )


def build_model(checkpoint: Dict, target_metric: str, node_dim: int, code_dim: Optional[int], graph_pooling: str) -> SimpleDifferentialGNN:
    args_dict = checkpoint.get("args", {})
    return SimpleDifferentialGNN(
        node_dim=node_dim,
        hidden_dim=args_dict.get("hidden_dim", 128),
        num_layers=args_dict.get("num_layers", 2),
        dropout=args_dict.get("dropout", 0.1),
        target_metric=target_metric,
        differential=args_dict.get("differential", True),
        gnn_type=args_dict.get("gnn_type", "gcn"),
        kernel_baseline=args_dict.get("kernel_baseline", "learned"),
        use_code_feature=bool(args_dict.get("use_code_feature", False)),
        code_dim=code_dim,
        graph_pooling=graph_pooling,
    )


def split_indices(n: int, seed: int) -> Dict[str, torch.Tensor]:
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(n, generator=g)
    train_size = int(0.8 * n)
    valid_size = int(0.1 * n)
    return {
        "train": indices[:train_size],
        "valid": indices[train_size : train_size + valid_size],
        "test": indices[train_size + valid_size :],
    }


def run_split(pairs: List[Dict], model: SimpleDifferentialGNN, device: torch.device, batch_size: int, target_metric: str):
    dataset = E2EDifferentialDataset(pairs, target_metric)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=differential_collate_fn, num_workers=0)
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            kernel_graph = batch["kernel_graph"].to(device)
            design_graph = batch["design_graph"].to(device)
            pragma_count = batch["pragma_count"].to(device)
            design_code = batch.get("design_code")
            if design_code is not None:
                design_code = design_code.to(device)
            outputs = model(kernel_graph, design_graph, pragma_count, design_code)
            pred = outputs["design_pred"].view(-1)
            true = batch["design_perf"].to(device).view(-1)
            preds.append(pred.cpu())
            trues.append(true.cpu())
    preds = torch.cat(preds)
    trues = torch.cat(trues)
    mae = torch.mean(torch.abs(preds - trues)).item()
    mask = trues != 0
    if mask.any():
        mape = (torch.mean(torch.abs(preds[mask] - trues[mask]) / torch.abs(trues[mask])) * 100).item()
    else:
        mape = float("nan")
    rmse = torch.sqrt(torch.mean((preds - trues) ** 2)).item()
    detail = []
    for p, t, pair in zip(preds, trues, pairs):
        if torch.abs(t).item() > 1e-8:
            sample_mape = float(torch.abs(p - t) / torch.abs(t) * 100)
        else:
            sample_mape = float("nan")
        detail.append(
            {
                "pair_id": pair["pair_id"],
                "kernel": pair["design_info"]["algo_name"],
                "design_base_path": pair["design_info"].get("base_path"),
                "design_true": float(t),
                "design_pred": float(p),
                "design_mae": float(abs(p - t)),
                "design_mape": sample_mape,
            }
        )
    return {"design_mae": mae, "design_mape": mape, "design_rmse": rmse}, detail


def _collect_pragma_combinations(design_path: Optional[str]) -> str:
    """简要汇总设计中的 pragma 组合（pipeline on/off、unroll 因子、array partition 因子）。"""
    if not design_path:
        return "pipeline=off; unroll=none; array_partition=none"
    design_dir = Path(design_path)
    if not design_dir.exists():
        return "pipeline=off; unroll=none; array_partition=none"

    pipeline_found = False
    pipeline_ii: Set[str] = set()
    unroll_factors: Set[str] = set()
    array_factors: Set[str] = set()

    src_files = []
    for ext in (".c", ".cpp", ".h", ".hpp"):
        src_files.extend(design_dir.rglob(f"*{ext}"))

    pipeline_re = re.compile(r"#pragma\s+HLS\s+PIPELINE(?:[^\\n]*?II\s*=\s*(\d+))?", re.IGNORECASE)
    unroll_re = re.compile(r"#pragma\s+HLS\s+UNROLL(?:[^\\n]*?factor\s*=\s*(\d+))?", re.IGNORECASE)
    array_re = re.compile(r"#pragma\s+HLS\s+ARRAY_PARTITION(?:[^\\n]*?factor\s*=\s*(\d+))?", re.IGNORECASE)

    for src in src_files:
        try:
            text = src.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for match in pipeline_re.finditer(text):
            pipeline_found = True
            ii_val = match.group(1)
            if ii_val:
                pipeline_ii.add(ii_val)
        for match in unroll_re.finditer(text):
            factor = match.group(1)
            if factor:
                unroll_factors.add(factor)
            else:
                unroll_factors.add("on")
        for match in array_re.finditer(text):
            factor = match.group(1)
            if factor:
                array_factors.add(factor)
            else:
                array_factors.add("on")

    def _sort_tokens(tokens: Set[str]) -> List[str]:
        return sorted(tokens, key=lambda v: (0, int(v)) if v.isdigit() else (1, v))

    pipeline_part = "pipeline=on" if pipeline_found else "pipeline=off"
    if pipeline_found and pipeline_ii:
        pipeline_part += f" (II={'/'.join(_sort_tokens(pipeline_ii))})"

    unroll_part = "unroll=" + ("/".join(_sort_tokens(unroll_factors)) if unroll_factors else "none")
    array_part = "array_partition=" + ("/".join(_sort_tokens(array_factors)) if array_factors else "none")

    return f"{pipeline_part}; {unroll_part}; {array_part}"


def _apply_hard_filter(
    pairs: List[Dict],
    target_metric: str,
    thresholds: Dict[str, tuple],
) -> List[Dict]:
    """按 p05-p95 阈值过滤样本；优先使用模型保存的 raw 阈值。"""
    if not pairs:
        return pairs
    metric_idx = METRIC_INDEX[target_metric]
    delta_key = f"{target_metric}_delta"

    local_thresholds = thresholds
    if not local_thresholds:
        # fallback: 基于当前样本计算阈值
        stats = {k: _compute_basic_stats(v) for k, v in _extract_metric_values(pairs, metric_idx, delta_key).items()}
        local_thresholds = {k: (v["p05"], v["p95"]) for k, v in stats.items() if v}

    if not local_thresholds:
        return pairs

    filtered = []
    for pair in pairs:
        vals = _extract_metric_values([pair], metric_idx, delta_key)
        keep = True
        for key, (low, high) in local_thresholds.items():
            value = vals[key][0]
            if value < low or value > high:
                keep = False
                break
        if keep:
            filtered.append(pair)
    return filtered


def main():
    parser = argparse.ArgumentParser(description="Train split (8:1:1) error analysis")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--design_base_dir", required=True)
    parser.add_argument("--kernel_base_dir", required=True)
    parser.add_argument("--cache_root", default="./graph_cache")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model_path, map_location=device)
    ckpt_args = checkpoint.get("args", {})
    target_metric = canonical_metric_name(ckpt_args.get("target_metric", "dsp"))
    use_code_feature = bool(ckpt_args.get("use_code_feature", False))
    code_model_path = ckpt_args.get("code_model_path")
    graph_pooling = ckpt_args.get("graph_pooling", "sum")

    default_out_dir = Path(args.output_dir) if args.output_dir else Path("output/train_split_error_analysis")
    out_dir = default_out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载训练阶段的 normalizer / 阈值，保证与训练流程一致
    model_dir = Path(args.model_path).resolve().parent
    stats_path = model_dir / "target_metric_stats.json"
    normalizers = None
    thresholds = {}
    if stats_path.exists():
        try:
            stats = json.load(open(stats_path, "r"))
            norm = stats.get("normalizer", {})
            if norm:
                normalizers = norm
            raw = stats.get("raw", {})
            for key in ("kernel", "design", "delta"):
                if key in raw and "p05" in raw[key] and "p95" in raw[key]:
                    thresholds[key] = (raw[key]["p05"], raw[key]["p95"])
        except Exception:
            pass

    if use_code_feature and not code_model_path:
        raise ValueError("use_code_feature=True 但未提供 code_model_path")

    processor = E2EDifferentialProcessor(
        kernel_base_dir=args.kernel_base_dir,
        design_base_dir=args.design_base_dir,
        output_dir=str(out_dir),
        cache_root=args.cache_root,
        rebuild_cache=False,
        hierarchical=str(ckpt_args.get("hierarchical", "off")).lower() == "on",
        region=str(ckpt_args.get("region", "off")).lower() == "on",
        max_workers=ckpt_args.get("max_workers", 32),
        use_code_feature=use_code_feature,
        code_model_path=code_model_path,
        code_cache_root=args.cache_root,
        code_pooling=ckpt_args.get("code_pooling", "last_token"),
        code_max_length=ckpt_args.get("code_max_length", 2048),
        code_normalize=bool(ckpt_args.get("code_normalize", False)),
        code_batch_size=ckpt_args.get("code_batch_size", 2),
        graph_pooling=graph_pooling,
        filter_resource_mismatch=str(ckpt_args.get("filter_resource_mismatch", "false")).lower() == "true",
    )

    pairs = [p for p in processor.collect_all_data() if p]
    if use_code_feature:
        pairs = processor.attach_code_features(pairs)
        before = len(pairs)
        pairs = [p for p in pairs if p.get("design_code_embedding") is not None]
        if len(pairs) < before:
            print(f"[Filter] 移除缺少 design_code_embedding 的样本 {before - len(pairs)} 条，剩余 {len(pairs)}")
    pairs = [
        p
        for p in pairs
        if p
        and p.get("kernel_graph") is not None
        and p.get("design_graph") is not None
        and p["kernel_graph"].y is not None
        and p["design_graph"].y is not None
    ]
    before_filter = len(pairs)
    pairs = _apply_hard_filter(pairs, target_metric, thresholds)
    after_filter = len(pairs)
    if before_filter != after_filter:
        print(f"[Filter] 应用 p05-p95 硬过滤（使用模型统计阈值优先），移除 {before_filter - after_filter} 条，剩余 {after_filter}")
    if not pairs:
        raise RuntimeError("未找到有效配对样本")

    splits = split_indices(len(pairs), args.seed)
    subsets = {name: [pairs[int(i)] for i in idx] for name, idx in splits.items()}

    sample_pair = pairs[0]
    node_dim = sample_pair["kernel_graph"].x.size(1)
    code_dim = None
    if use_code_feature:
        code_sample = sample_pair.get("design_code_embedding")
        if code_sample is None:
            raise ValueError("use_code_feature=True 但样本缺少 design_code_embedding")
        code_dim = code_sample.shape[-1]

    model = build_model(checkpoint, target_metric, node_dim, code_dim, graph_pooling).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_subset = subsets["test"]
    test_metrics, test_detail = run_split(test_subset, model, device, args.batch_size, target_metric)

    # 为 top10 报告补充 pragma 组合信息
    for entry in test_detail:
        entry["pragma_summary"] = _collect_pragma_combinations(entry.get("design_base_path"))

    pd.DataFrame(test_detail).to_csv(out_dir / "test_predictions.csv", index=False)

    # test 集统计：按 kernel 聚合、top10 MAPE
    test_df = pd.DataFrame(test_detail)
    kernel_stats = (
        test_df.groupby("kernel")
        .agg(
            design_mae=("design_mae", "mean"),
            design_mape=("design_mape", lambda x: np.nanmean(x)),
            count=("design_mae", "size"),
        )
        .sort_values("design_mape", ascending=False)
    )
    kernel_stats.to_csv(out_dir / "test_kernel_impact.csv")
    top10 = test_df.dropna(subset=["design_mape"]).sort_values("design_mape", ascending=False).head(10)
    top10.to_csv(out_dir / "test_design_top10_mape.csv", index=False)

    summary = {
        "total_pairs": len(pairs),
        "splits": {"test": len(test_subset)},
        "metrics": {"test": test_metrics},
        "kernel_stats_csv": str(out_dir / "test_kernel_impact.csv"),
        "top_designs_csv": str(out_dir / "test_design_top10_mape.csv"),
    }
    with open(out_dir / "train_split_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # log
    log_lines = [
        f"Total pairs: {len(pairs)}",
        f"Split sizes: {{'test': {len(test_subset)}}}",
        "Test metrics (MAE/MAPE/RMSE):",
    ]
    m = test_metrics
    mape_str = f"{m['design_mape']:.6f}" if not np.isnan(m["design_mape"]) else "nan"
    log_lines.append(f"  test: MAE={m['design_mae']:.6f}, MAPE={mape_str}, RMSE={m['design_rmse']:.6f}")
    log_lines.append("\n[Test kernel impact]")
    for _, row in kernel_stats.reset_index().iterrows():
        mape_str = f"{row.design_mape:.6f}" if not np.isnan(row.design_mape) else "nan"
        log_lines.append(f"  {row['kernel']}: MAE={row.design_mae:.6f}, MAPE={mape_str}, n={int(row['count'])}")
    log_lines.append("\n[Test top10 MAPE designs]")
    for _, r in top10.iterrows():
        log_lines.append(
            f"  {r['pair_id']}: MAPE={r['design_mape']:.3f}, MAE={r['design_mae']:.3f}, kernel={r['kernel']}, true={r['design_true']:.3f}, pred={r['design_pred']:.3f}, pragma={r.get('pragma_summary','')}"
        )
    with open(out_dir / "train_split_analysis.log", "w") as f:
        f.write("\n".join(log_lines))

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
