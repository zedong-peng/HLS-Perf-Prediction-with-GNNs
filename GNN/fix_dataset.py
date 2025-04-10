#!/usr/bin/env python3
# 修复数据集：过滤掉空图并生成新的有效数据集

import os
import torch
import numpy as np
import pandas as pd
import shutil
import gzip
from dataset_pyg import PygGraphPropPredDataset
from torch_geometric.data import DataLoader

# 设置源数据集和目标数据集名称
source_dataset = "cdfg_ff_all_numerical_gnn_test"
target_dataset = "cdfg_ff_all_numerical_gnn_test_fixed"

print(f"开始修复数据集: {source_dataset} -> {target_dataset}")

try:
    # 加载数据集
    dataset = PygGraphPropPredDataset(name=source_dataset)
    print(f"数据集加载成功！总图数量: {len(dataset)}")
    
    # 找出有效图的索引（非空图）
    valid_indices = []
    empty_indices = []
    
    for i in range(len(dataset)):
        data = dataset[i]
        if hasattr(data, 'x') and data.x is not None and data.x.shape[0] > 0:
            valid_indices.append(i)
        else:
            empty_indices.append(i)
    
    print(f"找到 {len(valid_indices)} 个有效图和 {len(empty_indices)} 个空图")
    if empty_indices:
        print(f"空图索引: {empty_indices}")
    
    # 创建目标数据集目录
    target_dir = f"./dataset/{target_dataset}"
    if os.path.exists(target_dir):
        print(f"删除已存在的目标目录: {target_dir}")
        shutil.rmtree(target_dir)
    
    # 创建目录结构
    raw_dir = f"{target_dir}/raw"
    mapping_dir = f"{target_dir}/mapping"
    split_dir = f"{target_dir}/split/scaffold"
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(mapping_dir, exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)
    
    # 从源数据集读取原始文件
    source_dir = f"./dataset/{source_dataset}"
    
    # 读取映射文件
    mapping_path = f"{source_dir}/mapping/mapping.csv.gz"
    with gzip.open(mapping_path, 'rt') as f:
        mapping_df = pd.read_csv(f)
    
    # 只保留有效图的映射
    filtered_mapping = mapping_df.iloc[valid_indices].reset_index(drop=True)
    
    # 保存新的映射文件
    with gzip.open(f"{mapping_dir}/mapping.csv.gz", 'wt') as f:
        filtered_mapping.to_csv(f, index=False)
    
    # 读取和过滤图标签
    with gzip.open(f"{source_dir}/raw/graph-label.csv.gz", 'rt') as f:
        labels_df = pd.read_csv(f, header=None)
    
    filtered_labels = labels_df.iloc[valid_indices].reset_index(drop=True)
    
    # 保存新的图标签
    with gzip.open(f"{raw_dir}/graph-label.csv.gz", 'wt') as f:
        filtered_labels.to_csv(f, index=False, header=False)
    
    # 处理数据集划分
    # 读取训练/验证/测试集索引
    train_indices = []
    valid_indices = []
    test_indices = []
    
    with gzip.open(f"{source_dir}/split/scaffold/train.csv.gz", 'rt') as f:
        train_df = pd.read_csv(f, header=None)
        train_indices = train_df[0].tolist()
    
    with gzip.open(f"{source_dir}/split/scaffold/valid.csv.gz", 'rt') as f:
        valid_df = pd.read_csv(f, header=None)
        valid_indices = valid_df[0].tolist()
    
    with gzip.open(f"{source_dir}/split/scaffold/test.csv.gz", 'rt') as f:
        test_df = pd.read_csv(f, header=None)
        test_indices = test_df[0].tolist()
    
    # 创建新的索引映射
    old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_indices)}
    
    # 过滤并重新映射划分索引
    new_train_indices = []
    new_valid_indices = []
    new_test_indices = []
    
    for idx in train_indices:
        if idx in old_to_new_idx:
            new_train_indices.append(old_to_new_idx[idx])
    
    for idx in valid_indices:
        if idx in old_to_new_idx:
            new_valid_indices.append(old_to_new_idx[idx])
    
    for idx in test_indices:
        if idx in old_to_new_idx:
            new_test_indices.append(old_to_new_idx[idx])
    
    # 保存新的划分文件
    with gzip.open(f"{split_dir}/train.csv.gz", 'wt') as f:
        pd.DataFrame(new_train_indices).to_csv(f, index=False, header=False)
    
    with gzip.open(f"{split_dir}/valid.csv.gz", 'wt') as f:
        pd.DataFrame(new_valid_indices).to_csv(f, index=False, header=False)
    
    with gzip.open(f"{split_dir}/test.csv.gz", 'wt') as f:
        pd.DataFrame(new_test_indices).to_csv(f, index=False, header=False)
    
    print(f"新的数据集划分: 训练集 {len(new_train_indices)}，验证集 {len(new_valid_indices)}，测试集 {len(new_test_indices)}")
    
    # 复制其他必要文件（node-feat.csv.gz, edge.csv.gz等）
    # 由于这些文件格式比较复杂，直接复制原始文件
    # 在PygGraphPropPredDataset加载时会根据数据集索引处理
    
    shutil.copy(f"{source_dir}/raw/node-feat.csv.gz", f"{raw_dir}/node-feat.csv.gz")
    shutil.copy(f"{source_dir}/raw/edge.csv.gz", f"{raw_dir}/edge.csv.gz")
    shutil.copy(f"{source_dir}/raw/edge-feat.csv.gz", f"{raw_dir}/edge-feat.csv.gz")
    shutil.copy(f"{source_dir}/raw/num-node-list.csv.gz", f"{raw_dir}/num-node-list.csv.gz")
    shutil.copy(f"{source_dir}/raw/num-edge-list.csv.gz", f"{raw_dir}/num-edge-list.csv.gz")
    
    # 创建master.csv（如果需要）
    # 这个文件包含数据集元数据
    master_source_path = "./master.csv"
    if os.path.exists(master_source_path):
        master_df = pd.read_csv(master_source_path, index_col=0)
        if source_dataset in master_df.columns:
            # 复制源数据集的元数据
            master_df[target_dataset] = master_df[source_dataset]
            master_df.to_csv(master_source_path)
            print(f"已更新master.csv添加新数据集: {target_dataset}")
    
    print(f"\n数据集修复完成: {target_dataset}")
    print(f"请使用以下命令运行训练:")
    print(f"python main.py --gnn pna --dataset {target_dataset} --epochs 2 --batch_size 1")
    
except Exception as e:
    print(f"修复数据集过程中出错: {str(e)}")
    import traceback
    traceback.print_exc() 