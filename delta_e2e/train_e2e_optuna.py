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
import os
import gc
from typing import List, Dict
from train_e2e import (
    E2EDifferentialProcessor,
    AVAILABLE_RESOURCES,
    SimpleDifferentialGNN,
    E2EDifferentialDataset,
    train_epoch,
    evaluate_model,
    differential_collate_fn
)
from torch_geometric.data import DataLoader
from utils import (
    parse_xml_into_graph_single, node_to_feature_vector, edge_to_feature_vector,
    get_node_feature_dims, get_edge_feature_dims
)


# 全局缓存，避免每个trial重复构建数据
TRAIN_PAIRS: List[Dict] = []
VAL_PAIRS: List[Dict] = []
NODE_DIM: int = -1


def prepare_data(max_pairs: int = None, target_metric: str = 'ff'):
    """一次性准备数据集与划分，供所有 Optuna trial 复用"""
    global TRAIN_PAIRS, VAL_PAIRS, NODE_DIM

    # 数据路径（与仓库规则对齐）
    kernel_base_dir = '/home/user/zedongpeng/workspace/Huggingface/forgehls_kernels/'
    design_base_dir = '/home/user/zedongpeng/workspace/Huggingface/forgehls_lite_100designs/'
    output_dir = './output_optuna'
    cache_root = './graph_cache'

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_root, exist_ok=True)

    processor = E2EDifferentialProcessor(
        kernel_base_dir=kernel_base_dir,
        design_base_dir=design_base_dir,
        output_dir=output_dir,
        cache_root=cache_root,
        rebuild_cache=False
    )
    pairs = processor.collect_all_data()
    if not pairs:
        raise RuntimeError('未找到任何有效配对数据，检查数据路径是否正确。')

    # 可选：限制配对数量，加速搜索
    if max_pairs is not None and len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]

    # 划分训练/验证集（固定随机种子，确保各trial一致）
    rng = np.random.RandomState(42)
    idxs = rng.permutation(len(pairs))
    split = int(0.8 * len(pairs))
    TRAIN_PAIRS = [pairs[i] for i in idxs[:split]]
    VAL_PAIRS = [pairs[i] for i in idxs[split:]]

    # 确定 node_dim
    if len(TRAIN_PAIRS) == 0:
        raise ValueError('训练数据为空，无法确定node_dim')
    NODE_DIM = TRAIN_PAIRS[0]['kernel_graph'].x.shape[1]

    # 轻量的 CUDA 性能优化
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


def objective(trial):
    # 超参数搜索空间
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 2, 4)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32])
    gnn_type = trial.suggest_categorical('gnn_type', ['gin'])
    # target_metric建议在命令行或脚本顶部指定，每次只测一个
    target_metric = 'lut'  # 或 dsp 'lut'、'ff'，可手动切换
    differential = True  # 只做差分预测

    # 低保真策略：限制每个trial的训练轮数，并设置早停
    max_epochs = trial.suggest_int('epochs', 10, 30)
    early_stop_patience = 5

    # 复用全局数据
    if len(TRAIN_PAIRS) == 0 or len(VAL_PAIRS) == 0:
        raise RuntimeError('全局数据未准备，请先调用 prepare_data()')
    train_pairs = TRAIN_PAIRS
    val_pairs = VAL_PAIRS

    # 构建Dataset和DataLoader
    # 获取真实的node_dim
    node_dim = NODE_DIM
    train_dataset = E2EDifferentialDataset(train_pairs, target_metric=target_metric)
    val_dataset = E2EDifferentialDataset(val_pairs, target_metric=target_metric)
    num_workers = 4 if os.cpu_count() and os.cpu_count() >= 4 else 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=differential_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=differential_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
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
        gnn_type=gnn_type
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 训练与验证
    best_val = float('inf')
    no_improve = 0
    for epoch in range(max_epochs):
        train_loss = train_epoch(model, device, train_loader, optimizer)
        val_metrics = evaluate_model(model, val_loader, device, target_metric=target_metric)
        val_mae = val_metrics['delta_mae']
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
    # 仅显示重要日志
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # 一次性准备数据（可选：限制用于搜索的样本对数量，加速）
    prepare_data(target_metric='ff')
    # 使用更激进的裁剪器 + TPE 采样器；最小化验证 MAE
    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=5, reduction_factor=3)
    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True)
    study = optuna.create_study(direction='minimize', pruner=pruner, sampler=sampler)
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    print('Best trial:')
    print(study.best_trial)
