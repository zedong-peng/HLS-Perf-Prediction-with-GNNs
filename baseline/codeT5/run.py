from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch

from .config import CodeT5Config
from .trainer import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CodeT5-only QoR baseline")
    parser.add_argument("--metric", choices=["lut", "ff", "dsp", "all"], default="all")
    parser.add_argument("--design-root", default="/home/user/zedongpeng/workspace/Huggingface/forgehls_lite_100designs")
    parser.add_argument("--ood-design-root", default=None)
    parser.add_argument("--cache-dir", default="baseline/codeT5/cache")
    parser.add_argument("--output-dir", default="baseline/codeT5/artifacts")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--max-designs", type=int, default=None)
    parser.add_argument("--max-ood-designs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--mlp-hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--tokenizer", default="Salesforce/codet5-small")
    parser.add_argument("--encoder", default="Salesforce/codet5-small")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--no-swanlab", action="store_true")
    parser.add_argument("--swan-project", default="CodeT5-Baseline")
    parser.add_argument("--swan-prefix", default="codet5")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics: List[str] = ["lut", "ff", "dsp"] if args.metric == "all" else [args.metric]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    summaries: Dict[str, Dict[str, object]] = {}
    for metric in metrics:
        config = CodeT5Config(
            metric_name=metric,
            design_root=args.design_root,
            ood_design_root=args.ood_design_root,
            cache_dir=Path(args.cache_dir),
            output_dir=Path(args.output_dir),
            dataset_name=Path(args.design_root).name,
            ood_dataset_name=Path(args.ood_design_root).name if args.ood_design_root else None,
            rebuild_cache=args.rebuild_cache,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            min_delta=args.min_delta,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            max_designs=args.max_designs,
            max_ood_designs=args.max_ood_designs,
            num_workers=args.num_workers,
            device=device,
            hidden_dim=args.hidden_dim,
            mlp_hidden=args.mlp_hidden,
            dropout=args.dropout,
            tokenizer_name=args.tokenizer,
            encoder_name=args.encoder,
            max_tokens=args.max_tokens,
            no_swanlab=args.no_swanlab,
            swan_project=args.swan_project,
            swan_prefix=args.swan_prefix,
        )

        summary = run_training(config)
        summaries[metric] = summary

    if len(summaries) > 1:
        print("Finished all metrics.")


if __name__ == "__main__":
    main()
