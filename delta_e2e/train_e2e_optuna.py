#!/usr/bin/env python3
"""
Optuna 超参数搜索脚本
=====================

本脚本基于 train_e2e.py 和 utils.py，使用 Optuna 自动搜索 GNN 训练的最佳超参数。

Author: Zedong Peng
"""

import optuna
import torch
import numpy as np
import torch.nn.functional as F
import os
import gc
import argparse
from typing import List, Dict, Optional
from train_e2e import (
    E2EDifferentialProcessor,
    AVAILABLE_RESOURCES,
    SimpleDifferentialGNN,
    E2EDifferentialDataset,
    train_epoch,
    evaluate_model,
    differential_collate_fn,
    RobustScaler,
    _compute_basic_stats,
    _extract_metric_values,
    _fit_robust_scaler
)
from torch_geometric.data import DataLoader
from torch_geometric.utils import degree
from utils import (
    parse_xml_into_graph_single, node_to_feature_vector, edge_to_feature_vector,
    get_node_feature_dims, get_edge_feature_dims
)


GLOBAL_DATA = {
    "train_pairs": [],
    "val_pairs": [],
    "node_dim": -1,
    "pna_deg": None,
    "edge_dim": 0,
    "code_dim": None,
    "normalizers": None,
}
CFG = {}


def prepare_data(
    kernel_base_dir: str,
    design_base_dir: str,
    output_dir: str,
    cache_root: str,
    target_metric: str,
    max_pairs: Optional[int] = None,
    hierarchical: bool = False,
    region: bool = False,
    use_code_feature: bool = False,
    code_model_path: Optional[str] = None,
    code_pooling: str = "last_token",
    code_max_length: int = 1024,
    code_normalize: bool = True,
    code_cache_root: Optional[str] = None,
    code_batch_size: int = 8,
    apply_hard_filter: bool = True,
    normalize_targets: bool = True,
):
    """一次性准备数据集与划分，供所有 Optuna trial 复用"""
    global GLOBAL_DATA

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_root, exist_ok=True)

    processor = E2EDifferentialProcessor(
        kernel_base_dir=kernel_base_dir,
        design_base_dir=design_base_dir,
        output_dir=output_dir,
        cache_root=cache_root,
        rebuild_cache=False,
        hierarchical=hierarchical,
        region=region,
        use_code_feature=use_code_feature,
        code_model_path=code_model_path,
        code_pooling=code_pooling,
        code_max_length=code_max_length,
        code_normalize=code_normalize,
        code_cache_root=code_cache_root,
        code_batch_size=code_batch_size,
        max_workers=8
    )
    pairs = processor.collect_all_data()
    if not pairs:
        raise RuntimeError('未找到任何有效配对数据，检查数据路径与缓存配置。')

    # 统计与过滤
    metric_idx = {'dsp': 0, 'lut': 1, 'ff': 2, 'latency': 3}[target_metric]
    delta_key = f'{target_metric}_delta' if target_metric != 'latency' else 'latency_delta'
    raw_stats = {k: _compute_basic_stats(v) for k, v in _extract_metric_values(pairs, metric_idx, delta_key).items()}
    if apply_hard_filter:
        thresholds = {k: (v["p05"], v["p95"]) for k, v in raw_stats.items() if v}
        filtered_pairs: List[Dict] = []
        for pair in pairs:
            vals = _extract_metric_values([pair], metric_idx, delta_key)
            keep = True
            for key, (low, high) in thresholds.items():
                value = vals[key][0]
                if value < low or value > high:
                    keep = False
                    break
            if keep:
                filtered_pairs.append(pair)
        if not filtered_pairs:
            raise ValueError("硬过滤后无有效样本，考虑关闭 --apply_hard_filter 或调整数据。")
        pairs = filtered_pairs

    # 可选：限制配对数量，加速搜索
    if max_pairs is not None and len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]

    # 划分训练/验证集（固定随机种子，确保各trial一致）
    rng = np.random.RandomState(42)
    idxs = rng.permutation(len(pairs))
    split = int(0.8 * len(pairs))
    train_pairs = [pairs[i] for i in idxs[:split]]
    val_pairs = [pairs[i] for i in idxs[split:]]

    # 确定 node_dim
    if len(train_pairs) == 0:
        raise ValueError('训练数据为空，无法确定node_dim')
    node_dim = train_pairs[0]['kernel_graph'].x.shape[1]
    edge_dim = 0
    sample_edge = train_pairs[0]['kernel_graph'].edge_attr
    if sample_edge is not None:
        edge_dim = sample_edge.size(1)
    elif train_pairs[0]['design_graph'].edge_attr is not None:
        edge_dim = train_pairs[0]['design_graph'].edge_attr.size(1)

    code_dim = None
    if use_code_feature:
        sample_code = train_pairs[0].get('design_code_embedding')
        if sample_code is None:
            raise ValueError("use_code_feature=True 但样本缺少 design_code_embedding")
        code_dim = sample_code.shape[-1]

    def _compute_pna_degree_histogram(pairs_subset: List[Dict]) -> torch.Tensor:
        hist = torch.zeros(1, dtype=torch.long)
        for pair in pairs_subset:
            for graph in (pair['kernel_graph'], pair['design_graph']):
                if not hasattr(graph, 'edge_index') or graph.edge_index is None:
                    continue
                deg = degree(graph.edge_index[1], num_nodes=graph.num_nodes, dtype=torch.long)
                if deg.numel() == 0:
                    continue
                max_deg = int(deg.max().item())
                if hist.numel() <= max_deg:
                    hist = torch.cat([hist, torch.zeros(max_deg - hist.numel() + 1, dtype=torch.long)])
                hist[:max_deg + 1] += torch.bincount(deg, minlength=max_deg + 1)
        if hist.sum() == 0:
            hist[0] = 1
        return hist

    pna_deg = _compute_pna_degree_histogram(train_pairs)

    # 轻量的 CUDA 性能优化
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # 拟合稳健归一化器（基于训练集）
    normalizers = None
    if normalize_targets:
        train_values = _extract_metric_values(train_pairs, metric_idx, delta_key)
        if any(len(v) == 0 for v in train_values.values()):
            raise ValueError("训练集目标为空，无法拟合归一化器")
        normalizers = {k: _fit_robust_scaler(v) for k, v in train_values.items()}

    GLOBAL_DATA = {
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
        "node_dim": node_dim,
        "pna_deg": pna_deg,
        "edge_dim": edge_dim,
        "code_dim": code_dim,
        "normalizers": normalizers,
    }


def objective(trial):
    if not GLOBAL_DATA["train_pairs"]:
        raise RuntimeError('全局数据未准备，请先调用 prepare_data()')
    target_metric = CFG["target_metric"]
    differential = CFG.get("differential", True)
    use_code_feature = CFG.get("use_code_feature", False)
    code_dim = GLOBAL_DATA.get("code_dim")
    loss_fn = CFG.get("loss_fn", F.l1_loss)

    # 超参数搜索空间
    hidden_dim = trial.suggest_categorical('hidden_dim', [96, 128, 160, 192])
    num_layers = trial.suggest_int('num_layers', 2, 3)
    dropout = trial.suggest_float('dropout', 0.0, 0.2)
    lr = trial.suggest_float('lr', 3e-4, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32])
    gnn_type = 'pna'  # 与实验3保持一致

    # 低保真策略：限制每个trial的训练轮数，并设置早停
    max_epochs = trial.suggest_int('epochs', 20, 40)
    early_stop_patience = 6

    train_pairs = GLOBAL_DATA["train_pairs"]
    val_pairs = GLOBAL_DATA["val_pairs"]

    # 构建Dataset和DataLoader
    node_dim = GLOBAL_DATA["node_dim"]
    train_dataset = E2EDifferentialDataset(train_pairs, target_metric=target_metric)
    val_dataset = E2EDifferentialDataset(val_pairs, target_metric=target_metric)
    num_workers = 0  # 避免 code 特征与多进程冲突
    normalizers = GLOBAL_DATA.get("normalizers")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=differential_collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=differential_collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    # 构建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleDifferentialGNN(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        target_metric=target_metric,
        differential=differential,
        gnn_type=gnn_type,
        pna_deg=GLOBAL_DATA["pna_deg"],
        edge_dim=GLOBAL_DATA["edge_dim"],
        use_code_feature=use_code_feature,
        code_dim=code_dim
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 训练与验证
    best_val = float('inf')
    no_improve = 0
    for epoch in range(max_epochs):
        train_loss = train_epoch(
            model, device, train_loader, optimizer,
            loss_fn=loss_fn, normalizers=normalizers
        )
        val_metrics = evaluate_model(
            model, val_loader, device,
            target_metric=target_metric,
            loss_fn=loss_fn,
            normalizers=normalizers
        )
        # 以设计 MAE 作为优化目标（差分模式下更贴近最终指标）
        val_mae = val_metrics['design_mae']
        if val_mae < best_val:
            best_val = val_mae
            no_improve = 0
        else:
            no_improve += 1
        # Optuna 报告并按 pruner 策略裁剪
        trial.report(val_mae, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
        # 简单早停，加速不收敛的trial
        if no_improve >= early_stop_patience:
            break

    # 清理缓存，避免多trial OOM
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_val

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Optuna 超参搜索（对齐 exp3 配置）")
    parser.add_argument('--target_metric', type=str, default='ff', choices=['ff', 'dsp', 'lut', 'latency'])
    parser.add_argument('--kernel_base_dir', type=str, default='/home/user/zedongpeng/workspace/Huggingface/forgehls_kernels')
    parser.add_argument('--design_base_dir', type=str, default='/home/user/zedongpeng/workspace/Huggingface/forgehls_PolyBench_part_500designs')
    parser.add_argument('--output_dir', type=str, default='./output_optuna')
    parser.add_argument('--cache_root', type=str, default='./graph_cache')
    parser.add_argument('--max_pairs', type=int, default=4000, help='可选：限制样本对数量，加速搜索')
    parser.add_argument('--n_trials', type=int, default=50)
    parser.add_argument('--use_code_feature', type=str, default='true', choices=['true', 'false'])
    parser.add_argument('--code_model_path', type=str, default='/home/user/zedongpeng/workspace/GiT/zedong/Code-Verification/Qwen/Qwen2.5-Coder-1.5B-Instruct')
    parser.add_argument('--code_pooling', type=str, default='last_token', choices=['last_token', 'mean'])
    parser.add_argument('--code_max_length', type=int, default=1024)
    parser.add_argument('--code_normalize', type=str, default='true', choices=['true', 'false'])
    parser.add_argument('--code_cache_root', type=str, default='./graph_cache')
    parser.add_argument('--code_batch_size', type=int, default=8)
    parser.add_argument('--hierarchical', type=str, default='off', choices=['on', 'off'])
    parser.add_argument('--region', type=str, default='on', choices=['on', 'off'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--study_name', type=str, default=None)
    parser.add_argument('--loss_fn', type=str, default='l1', choices=['l1', 'smoothl1'])
    parser.add_argument('--apply_hard_filter', type=str, default='true', choices=['true', 'false'])
    parser.add_argument('--normalize_targets', type=str, default='true', choices=['true', 'false'])
    args = parser.parse_args()

    # 仅显示重要日志
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    use_code_feature = args.use_code_feature.lower() == 'true'
    code_normalize = args.code_normalize.lower() == 'true'
    hierarchical = args.hierarchical.lower() == 'on'
    region = args.region.lower() == 'on'
    apply_hard_filter = args.apply_hard_filter.lower() == 'true'
    normalize_targets = args.normalize_targets.lower() == 'true'
    loss_name = args.loss_fn.lower()
    if loss_name == 'l1':
        loss_fn = F.l1_loss
    elif loss_name == 'smoothl1':
        loss_fn = F.smooth_l1_loss
    else:
        raise ValueError(f"Unsupported loss_fn: {loss_name}")
    if use_code_feature and not args.code_model_path:
        raise ValueError("use_code_feature=true 时必须提供 --code_model_path")

    CFG = {
        "target_metric": args.target_metric,
        "differential": True,
        "loss_fn": loss_fn,
    }

    # 一次性准备数据（可选：限制用于搜索的样本对数量，加速）
    prepare_data(
        kernel_base_dir=args.kernel_base_dir,
        design_base_dir=args.design_base_dir,
        output_dir=args.output_dir,
        cache_root=args.cache_root,
        target_metric=args.target_metric,
        max_pairs=args.max_pairs,
        hierarchical=hierarchical,
        region=region,
        use_code_feature=use_code_feature,
        code_model_path=args.code_model_path,
        code_pooling=args.code_pooling,
        code_max_length=args.code_max_length,
        code_normalize=code_normalize,
        code_cache_root=args.code_cache_root,
        code_batch_size=args.code_batch_size,
        apply_hard_filter=apply_hard_filter,
        normalize_targets=normalize_targets,
    )

    # 使用更激进的裁剪器 + TPE 采样器；最小化验证 MAE
    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=5, reduction_factor=3)
    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True)
    study = optuna.create_study(
        direction='minimize',
        pruner=pruner,
        sampler=sampler,
        study_name=args.study_name,
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    print('Best trial:')
    print(study.best_trial)
