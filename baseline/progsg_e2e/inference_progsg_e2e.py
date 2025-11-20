"""Run inference for a trained ProgSG(-multimodal) model on a new dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch_geometric.loader import DataLoader

from code_features import CodeEncoderConfig, SourceCodeFeatureCache
from dataset import DesignGraphProcessor, assign_sample_indices, attach_code_features
from model import ProgSGMultimodalModel, ProgSGStyleModel, TARGET_NAMES
from train_progsg_e2e import (
    LOSS_TARGET_INDICES,
    DesignGraphDataset,
    evaluate,
    save_predictions,
    split_indices,
    data_loader_from_subset,
)


def _coerce(value: Any, expected_type: type, fallback: Any) -> Any:
    if value is None:
        return fallback
    if isinstance(value, expected_type):
        return value
    try:
        if expected_type is bool:
            if isinstance(value, str):
                return value.lower() in {"1", "true", "yes", "y"}
            return bool(value)
        return expected_type(value)
    except (ValueError, TypeError):
        return fallback


def _load_checkpoint(model_path: Path) -> Dict[str, Any]:
    checkpoint = torch.load(model_path, map_location="cpu")
    if "model_state" not in checkpoint:
        raise RuntimeError(
            "Checkpoint missing 'model_state'. Please provide a file saved by train_progsg_e2e.py."
        )
    required_fields = {"model_state", "target_mean", "target_std", "target_names"}
    missing = required_fields - set(checkpoint.keys())
    if missing:
        raise RuntimeError(f"Checkpoint missing required fields: {sorted(missing)}")
    return checkpoint


def _resolve_args(checkpoint_args: Dict[str, Any], key: str, default: Any, expected_type: type) -> Any:
    value = checkpoint_args.get(key, default)
    return _coerce(value, expected_type, default)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference for ProgSG-style models.")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to model_best.pt or model_last.pt.")
    parser.add_argument("--ood_design_base_dir", type=Path, required=True, help="Directory containing OOD designs.")
    parser.add_argument("--output_dir", type=Path, default=None, help="Directory for inference artifacts.")
    parser.add_argument("--cache_root", type=Path, default=None, help="Override cache root for inference runs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size used for inference.")
    parser.add_argument("--num_workers", type=int, default=None, help="Override dataloader workers.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on.",
    )
    parser.add_argument("--rebuild_cache", action="store_true", help="Force regeneration of cached graphs/code.")
    parser.add_argument(
        "--id_design_base_dir",
        type=Path,
        default=None,
        help="Optional path to rerun ID test using the training split. Defaults to the training dataset directory if available.",
    )
    parser.add_argument(
        "--skip_id_eval",
        action="store_true",
        help="Disable ID test evaluation even if the dataset path is available.",
    )
    args = parser.parse_args()

    checkpoint = _load_checkpoint(args.model_path)
    checkpoint_args: Dict[str, Any] = checkpoint.get("args", {})

    output_dir = args.output_dir or (args.model_path.parent / "inference_outputs")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    cache_root_default = checkpoint_args.get("cache_root", output_dir / "cache")
    cache_root = args.cache_root or Path(cache_root_default)
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    enable_code = checkpoint.get("model_type", "graph") == "multimodal"
    hierarchical = _resolve_args(checkpoint_args, "hierarchical", False, bool)
    region = _resolve_args(checkpoint_args, "region", False, bool)
    max_workers_default = _resolve_args(checkpoint_args, "num_workers", 0, int)
    batch_size_default = _resolve_args(checkpoint_args, "batch_size", 16, int)
    dropout = float(checkpoint_args.get("dropout", 0.1))
    hidden_dim = _resolve_args(checkpoint_args, "hidden_dim", 512, int)
    num_layers = _resolve_args(checkpoint_args, "num_layers", 8, int)
    heads = _resolve_args(checkpoint_args, "heads", 8, int)
    activation = checkpoint_args.get("activation", "elu")

    num_workers = args.num_workers if args.num_workers is not None else max_workers_default
    batch_size = args.batch_size if args.batch_size is not None else batch_size_default

    processor = DesignGraphProcessor(
        design_base_dir=args.ood_design_base_dir,
        output_dir=output_dir,
        cache_root=cache_root,
        rebuild_cache=args.rebuild_cache,
        hierarchical=hierarchical,
        region=region,
        include_code=enable_code,
    )
    samples = processor.collect_designs()
    if not samples:
        raise RuntimeError("No designs were parsed from the provided OOD directory.")

    code_cache = None
    code_embedding_dim: Optional[int] = None
    if enable_code:
        code_encoder_name = checkpoint_args.get("code_encoder_name", "Salesforce/codet5-small")
        code_max_length = _resolve_args(checkpoint_args, "code_max_length", 256, int)
        code_cache_namespace = checkpoint_args.get("code_cache_namespace", "codet5_small_v1")
        code_local_files_only = _resolve_args(checkpoint_args, "code_local_files_only", False, bool)
        code_cache_device = checkpoint_args.get("code_cache_device", "cpu")
        if code_cache_device.startswith("cuda") and not torch.cuda.is_available():
            code_cache_device = "cpu"

        code_config = CodeEncoderConfig(
            model_name_or_path=code_encoder_name,
            max_length=code_max_length,
            cache_namespace=code_cache_namespace,
            local_files_only=code_local_files_only,
            store_token_embeddings=True,
            store_pooled=True,
        )
        code_cache = SourceCodeFeatureCache(cache_root=cache_root / "code_features", config=code_config, device=code_cache_device)
        attach_code_features(samples, code_cache, encode=True)

        example_inputs = next((s.code_inputs for s in samples if s.code_inputs is not None), None)
        if example_inputs is None:
            raise RuntimeError("Model requires code features but none were generated for the OOD dataset.")
        if "token_embeddings" in example_inputs:
            code_embedding_dim = int(example_inputs["token_embeddings"].shape[1])
        elif "pooled_embedding" in example_inputs:
            code_embedding_dim = int(example_inputs["pooled_embedding"].shape[-1])
        else:
            raise RuntimeError("Unable to infer code embedding dimensionality from cached features.")

    assign_sample_indices(samples)
    dataset = DesignGraphDataset(samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if enable_code:
        code_transformer_layers = _resolve_args(checkpoint_args, "code_transformer_layers", 2, int)
        code_transformer_heads = _resolve_args(checkpoint_args, "code_transformer_heads", 4, int)
        code_fusion_mode = checkpoint_args.get("code_fusion_mode", "concat")
        code_node_interaction = _resolve_args(checkpoint_args, "code_node_interaction", False, bool)
        model = ProgSGMultimodalModel(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            activation=activation,
            code_embedding_dim=code_embedding_dim or hidden_dim,
            code_transformer_layers=code_transformer_layers,
            code_transformer_heads=code_transformer_heads,
            fusion_mode=code_fusion_mode,
            node_token_interaction=code_node_interaction,
            targets=checkpoint["target_names"],
        )
    else:
        model = ProgSGStyleModel(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            activation=activation,
            targets=checkpoint["target_names"],
        )

    model.load_state_dict(checkpoint["model_state"])
    device = torch.device(args.device)
    model.to(device)
    target_mean = checkpoint["target_mean"].to(device)
    target_std = checkpoint["target_std"].to(device)

    eval_output = evaluate(
        model,
        loader,
        device,
        target_mean,
        target_std,
        loss_indices=LOSS_TARGET_INDICES,
        debug_nan=False,
        dataset=dataset,
    )

    print("Inference Metrics:")
    for name, metric_dict in eval_output["metrics"].items():
        stats = ", ".join(f"{metric}={value:.6f}" for metric, value in metric_dict.items())
        print(f"  {name.upper()}: {stats}")

    save_predictions(output_dir, "ood_inference", eval_output, dataset)
    summary_path = output_dir / "metrics_inference.json"
    summary_payload = {
        "metrics": eval_output["metrics"],
        "loss": eval_output["loss"],
        "num_samples": int(eval_output["preds"].shape[0]),
        "model_path": str(args.model_path.resolve()),
        "ood_design_base_dir": str(args.ood_design_base_dir.resolve()),
        "target_names": checkpoint["target_names"],
    }
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, indent=2)
    print(f"Saved predictions and metrics to {output_dir}")

    if args.skip_id_eval:
        return

    id_base_dir = args.id_design_base_dir
    if id_base_dir is None:
        train_base_dir = checkpoint_args.get("design_base_dir")
        if train_base_dir:
            id_candidate = Path(str(train_base_dir))
            if id_candidate.exists():
                id_base_dir = id_candidate
    if id_base_dir is None or not Path(id_base_dir).exists():
        print("ID evaluation skipped: training dataset directory unavailable. Use --id_design_base_dir to specify it.")
        return

    print(f"\nRunning ID evaluation using dataset at {id_base_dir}...")
    train_seed = _resolve_args(checkpoint_args, "seed", 42, int)
    train_ratio = float(checkpoint_args.get("train_ratio", 0.8))
    valid_ratio = float(checkpoint_args.get("valid_ratio", 0.1))
    max_designs = checkpoint_args.get("max_designs")
    max_designs = None if max_designs in (None, "None") else _coerce(max_designs, int, None)

    id_processor = DesignGraphProcessor(
        design_base_dir=Path(id_base_dir),
        output_dir=output_dir,
        cache_root=cache_root / "id",
        rebuild_cache=args.rebuild_cache,
        hierarchical=hierarchical,
        region=region,
        include_code=enable_code,
    )
    id_samples = id_processor.collect_designs(max_designs=max_designs, seed=train_seed)
    if not id_samples:
        raise RuntimeError("ID evaluation requested but no samples were parsed from the training dataset directory.")

    if enable_code and code_cache is not None:
        attach_code_features(id_samples, code_cache, encode=True)
    assign_sample_indices(id_samples)
    id_dataset = DesignGraphDataset(id_samples)
    split = split_indices(len(id_dataset), train_ratio, valid_ratio, seed=train_seed)
    test_loader = data_loader_from_subset(
        id_dataset,
        split["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    id_eval = evaluate(
        model,
        test_loader,
        device,
        target_mean,
        target_std,
        loss_indices=LOSS_TARGET_INDICES,
        debug_nan=False,
        dataset=id_dataset,
    )
    print("ID Test Metrics (from training split):")
    for name, metric_dict in id_eval["metrics"].items():
        stats = ", ".join(f"{metric}={value:.6f}" for metric, value in metric_dict.items())
        print(f"  {name.upper()}: {stats}")
    save_predictions(output_dir, "id_inference", id_eval, id_dataset)
    summary_path_id = output_dir / "metrics_id_inference.json"
    summary_payload_id = {
        "metrics": id_eval["metrics"],
        "loss": id_eval["loss"],
        "num_samples": int(id_eval["preds"].shape[0]),
        "train_ratio": train_ratio,
        "valid_ratio": valid_ratio,
        "seed": train_seed,
        "target_names": checkpoint["target_names"],
    }
    with summary_path_id.open("w", encoding="utf-8") as fh:
        json.dump(summary_payload_id, fh, indent=2)
    print(f"Saved ID predictions and metrics to {summary_path_id.parent}")


if __name__ == "__main__":
    main()
