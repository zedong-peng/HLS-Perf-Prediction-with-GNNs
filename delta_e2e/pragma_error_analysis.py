#!/usr/bin/env python3
"""
Pragma-aware error dump (structured JSON).

For each design, emits predictions and a list of normalized pragma entries:
[
  { "id": 1, "type": "ARRAY_PARTITION", "factor": 4, "dim": 2, "variable": "D", "mode": "cyclic" },
  { "id": 2, "type": "PIPELINE", "mode": "on" },  # mode: on/off
  { "id": 3, "type": "UNROLL", "factor": 4 }
]
Only PIPELINE / UNROLL / ARRAY_PARTITION are captured.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import DataLoader

try:
    from delta_e2e.train_e2e import (  # type: ignore
        E2EDifferentialDataset,
        E2EDifferentialProcessor,
        SimpleDifferentialGNN,
        differential_collate_fn,
    )
    from delta_e2e.utils import get_edge_feature_dims  # type: ignore
except Exception:
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_e2e import (
        E2EDifferentialDataset,
        E2EDifferentialProcessor,
        SimpleDifferentialGNN,
        differential_collate_fn,
        get_edge_feature_dims,
    )


def parse_pragmas(design_dir: Path) -> Tuple[List[Dict], bool]:
    """Extract structured pragmas (PIPELINE/UNROLL/ARRAY_PARTITION) and has_pipeline flag."""
    pragmas: List[Dict] = []
    has_pipeline = False
    pipeline_pat = re.compile(r"#\s*pragma\s+HLS\s+PIPELINE\b([^\n]*)", re.IGNORECASE)
    unroll_pat = re.compile(r"#\s*pragma\s+HLS\s+UNROLL\b([^\n]*)", re.IGNORECASE)
    unroll_factor_pat = re.compile(r"factor\s*=\s*(\d+)", re.IGNORECASE)
    array_pat = re.compile(r"#\s*pragma\s+HLS\s+ARRAY_PARTITION\b([^\n]*)", re.IGNORECASE)
    array_factor_pat = re.compile(r"factor\s*=\s*(\d+)", re.IGNORECASE)
    array_type_pat = re.compile(r"(complete|cyclic|block)", re.IGNORECASE)
    array_dim_pat = re.compile(r"dim\s*=\s*(\d+)", re.IGNORECASE)
    array_var_pat = re.compile(r"variable\s*=\s*([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)

    pid = 1
    for src in design_dir.rglob("*"):
        if not src.is_file() or src.suffix.lower() not in {".c", ".cpp", ".h", ".hpp"}:
            continue
        try:
            text = src.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for m in pipeline_pat.finditer(text):
            payload = m.group(1)
            mode = "off" if re.search(r"\boff\b", payload, re.IGNORECASE) else "on"
            pragmas.append({"id": pid, "type": "PIPELINE", "mode": mode})
            if mode == "on":
                has_pipeline = True
            pid += 1

        for m in unroll_pat.finditer(text):
            payload = m.group(1)
            if re.search(r"\boff\b", payload, re.IGNORECASE):
                pragmas.append({"id": pid, "type": "UNROLL", "mode": "off"})
            else:
                factor = None
                fm = unroll_factor_pat.search(payload)
                if fm:
                    try:
                        factor = int(fm.group(1))
                    except Exception:
                        factor = None
                pragmas.append({"id": pid, "type": "UNROLL", "mode": "on", "factor": factor})
            pid += 1

        for m in array_pat.finditer(text):
            payload = m.group(1)
            factor = None
            dim = None
            mode = None
            var = None
            fm = array_factor_pat.search(payload)
            if fm:
                try:
                    factor = int(fm.group(1))
                except Exception:
                    factor = None
            dm = array_dim_pat.search(payload)
            if dm:
                try:
                    dim = int(dm.group(1))
                except Exception:
                    dim = None
            tm = array_type_pat.search(payload)
            if tm:
                mode = tm.group(1).lower()
            vm = array_var_pat.search(payload)
            if vm:
                var = vm.group(1)
            pragmas.append({"id": pid, "type": "ARRAY_PARTITION", "factor": factor, "dim": dim, "mode": mode, "variable": var})
            pid += 1

    return pragmas, has_pipeline


def build_model_from_checkpoint(
    checkpoint: Dict, target_metric: str, device: torch.device, dataset: E2EDifferentialDataset, graph_pooling: str
) -> Tuple[SimpleDifferentialGNN, Optional[int]]:
    args_dict = checkpoint.get("args", {})
    model_state = checkpoint["model_state_dict"]

    sample = dataset[0]
    node_dim = int(sample["design_graph"].x.size(1))

    edge_dim = None
    if "design_edge_encoder.weight" in model_state:
        edge_dim = model_state["design_edge_encoder.weight"].shape[1]
    else:
        edge_dims = get_edge_feature_dims()
        edge_dim = sum(edge_dims) if isinstance(edge_dims, list) else int(edge_dims)

    code_dim = None
    use_code_feature = bool(args_dict.get("use_code_feature", False))
    if use_code_feature:
        if "code_adapter.0.weight" not in model_state:
            raise ValueError("checkpoint缺少 code_adapter 权重，无法推断代码特征维度。")
        code_dim = model_state["code_adapter.0.weight"].shape[1]

    model = SimpleDifferentialGNN(
        node_dim=node_dim,
        hidden_dim=args_dict.get("hidden_dim", 128),
        num_layers=args_dict.get("num_layers", 2),
        target_metric=target_metric,
        differential=True,
        gnn_type=args_dict.get("gnn_type", "gcn"),
        use_code_feature=use_code_feature,
        code_dim=code_dim,
        graph_pooling=graph_pooling,
        kernel_baseline=args_dict.get("kernel_baseline", "learned"),
    ).to(device)
    model.load_state_dict(model_state, strict=False)
    model.eval()
    return model, code_dim


def main():
    parser = argparse.ArgumentParser(description="Pragma-aware error JSON dump")
    parser.add_argument("--model_path", required=True, help="Trained model checkpoint (.pt)")
    parser.add_argument("--design_base_dir", required=True, help="Design root dir")
    parser.add_argument("--kernel_base_dir", default=None, help="Kernel root dir (default from checkpoint)")
    parser.add_argument("--cache_root", default="./graph_cache_prag", help="Cache root for graphs")
    parser.add_argument("--code_cache_root", default=None, help="Override code cache root")
    parser.add_argument("--output_dir", default="./pragma_error_analysis", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--rebuild_cache", action="store_true", help="Force rebuild graph cache")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model_path, map_location=device)
    ckpt_args = checkpoint.get("args", {})
    target_metric = ckpt_args.get("target_metric", "dsp")
    use_code_feature = bool(ckpt_args.get("use_code_feature", False))
    code_model_path = ckpt_args.get("code_model_path")
    graph_pooling = ckpt_args.get("graph_pooling", "sum")

    processor = E2EDifferentialProcessor(
        kernel_base_dir=args.kernel_base_dir or ckpt_args.get("kernel_base_dir"),
        design_base_dir=args.design_base_dir,
        output_dir=args.output_dir,
        cache_root=args.cache_root,
        rebuild_cache=args.rebuild_cache,
        hierarchical=str(ckpt_args.get("hierarchical", "off")).lower() == "on",
        region=str(ckpt_args.get("region", "off")).lower() == "on",
        max_workers=ckpt_args.get("max_workers", 32),
        use_code_feature=use_code_feature,
        code_model_path=code_model_path,
        code_cache_root=args.code_cache_root or ckpt_args.get("code_cache_root"),
        code_pooling=ckpt_args.get("code_pooling", "last_token"),
        code_max_length=ckpt_args.get("code_max_length", 2048),
        code_normalize=bool(ckpt_args.get("code_normalize", False)),
        code_batch_size=ckpt_args.get("code_batch_size", 1),
        graph_pooling=graph_pooling,
    )

    pairs = processor.collect_all_data()
    if use_code_feature:
        pairs = processor.attach_code_features(pairs)
    if not pairs:
        try:
            rebuilt = processor._rebuild_index_from_pairs()
            if rebuilt > 0:
                pairs = processor._load_cached_pairs(materialize=True)
                if use_code_feature:
                    pairs = processor.attach_code_features(pairs)
        except Exception:
            pass
    if not pairs:
        raise RuntimeError("未找到有效配对")

    dataset = E2EDifferentialDataset(pairs, target_metric)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=differential_collate_fn)

    model, code_dim = build_model_from_checkpoint(checkpoint, target_metric, device, dataset, graph_pooling)

    records: List[Dict] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            kernel_batch = batch["kernel_graph"].to(device)
            design_batch = batch["design_graph"].to(device)
            pragma_count = batch["pragma_count"].to(device)
            design_code = batch.get("design_code")
            if design_code is not None:
                design_code = design_code.to(device)

            predictions = model(kernel_batch, design_batch, pragma_count, design_code)
            design_pred = predictions["design_pred"].squeeze()
            kernel_pred = predictions["kernel_pred"].squeeze()

            design_true = batch["design_perf"].to(device).squeeze()
            kernel_true = batch["kernel_perf"].to(device).squeeze()

            design_true_dn = design_true  # 数据集已使用原始尺度存储
            design_pred_dn = design_pred

            design_err = (design_pred_dn - design_true_dn).abs()

            for i in range(len(design_true)):
                pair_idx = batch_idx * args.batch_size + i
                pair = pairs[pair_idx]
                design_dir = Path(pair["design_info"]["base_path"])
                pragmas_struct, has_pipeline = parse_pragmas(design_dir)

                records.append(
                    {
                        "pair_id": pair["pair_id"],
                        "design_path": str(design_dir),
                        "design_true": float(design_true_dn[i].item()),
                        "design_pred": float(design_pred_dn[i].item()),
                        "design_mae": float(design_err[i].item()),
                        "design_mape": float(design_err[i] / (design_true_dn[i].abs() + 1e-8) * 100),
                        "has_pipeline": has_pipeline,
                        "pragmas": pragmas_struct,
                    }
                )

    out_path = os.path.join(args.output_dir, "pragma_error_results.json")
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"已输出 JSON: {out_path}，样本数 {len(records)}")

    # 统计日志输出
    import pandas as pd

    df = pd.DataFrame(records)
    overall_mae = df["design_mae"].mean()
    overall_mape = df["design_mape"].mean()
    # kernel 名从 path 中提取 algo 名
    def kernel_from_path(path: str) -> str:
        parts = Path(path).parts
        return parts[-2] if len(parts) >= 2 else "unknown"

    df["kernel"] = df["design_path"].apply(kernel_from_path)
    ker_stats = (
        df.groupby("kernel", as_index=False)
        .agg(design_mae=("design_mae", "mean"), design_mape=("design_mape", "mean"), count=("design_mae", "size"))
        .sort_values("design_mape", ascending=False)
    )
    top5 = df.sort_values("design_mape", ascending=False).head(5)

    log_lines = []
    log_lines.append(f"Total samples: {len(df)}")
    log_lines.append(f"Overall MAE: {overall_mae:.6f}, Overall MAPE: {overall_mape:.6f}")
    log_lines.append("\nPer-kernel MAE/MAPE:")
    for _, row in ker_stats.iterrows():
        log_lines.append(
            f"  {row['kernel']}: MAE={row.design_mae:.6f}, MAPE={row.design_mape:.6f}, n={int(row['count'])}"
        )
    log_lines.append("\nTop5 designs by MAPE:")
    for _, r in top5.iterrows():
        log_lines.append(
            f"  {r['pair_id']}: MAPE={r['design_mape']:.3f}, MAE={r['design_mae']:.3f}, path={r['design_path']}"
        )

    log_path = os.path.join(args.output_dir, "pragma_error_summary.log")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    print(f"已输出日志: {log_path}")


if __name__ == "__main__":
    main()
