#!/usr/bin/env python3
# 测试脚本：用于验证数据集完整性和有效性

import os
import torch
import numpy as np
from dataset_pyg import PygGraphPropPredDataset
from torch_geometric.data import DataLoader

# 设置要检查的数据集名称
dataset_name = "cdfg_ff_all_numerical_gnn_test"

print(f"开始验证数据集: {dataset_name}")

try:
    # 加载数据集
    dataset = PygGraphPropPredDataset(name=dataset_name)
    
    print(f"数据集加载成功！总图数量: {len(dataset)}")
    
    # 获取分割索引
    split_idx = dataset.get_idx_split()
    print(f"训练集大小: {len(split_idx['train'])}")
    print(f"验证集大小: {len(split_idx['valid'])}")
    print(f"测试集大小: {len(split_idx['test'])}")
    
    # 检查每个图
    empty_graphs = []
    small_graphs = []
    no_labels = []
    
    for i in range(len(dataset)):
        data = dataset[i]
        
        # 检查图是否为空
        if not hasattr(data, 'x') or data.x is None or data.x.shape[0] == 0:
            empty_graphs.append(i)
            continue
            
        # 检查图是否太小
        if data.x.shape[0] < 5:  # 节点数少于5个的图视为"小图"
            small_graphs.append(i)
            
        # 检查标签是否存在
        if not hasattr(data, 'y') or data.y is None:
            no_labels.append(i)
    
    if empty_graphs:
        print(f"警告：发现 {len(empty_graphs)} 个空图！")
        print(f"空图索引: {empty_graphs[:10]}...")
    else:
        print("所有图都包含节点和边，未发现空图。")
        
    if small_graphs:
        print(f"信息：发现 {len(small_graphs)} 个小图（节点数<5）")
        print(f"小图索引: {small_graphs[:10]}...")
    
    if no_labels:
        print(f"警告：发现 {len(no_labels)} 个无标签图！")
        print(f"无标签图索引: {no_labels[:10]}...")
    else:
        print("所有图都有有效标签。")
    
    # 创建和测试数据加载器
    print("\n测试数据加载器...")
    batch_size = 1
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # 尝试迭代数据加载器
    batch_count = 0
    empty_batch_count = 0
    problem_batches = []
    
    for i, batch in enumerate(loader):
        batch_count += 1
        
        if not hasattr(batch, 'x') or batch.x.shape[0] == 0 or not hasattr(batch, 'batch') or batch.batch.shape[0] == 0:
            empty_batch_count += 1
            problem_batches.append(i)
            print(f"警告：发现空批次 #{i}")
            continue
            

        print(f"批次 #{i} 信息:")
        print(f"  节点特征形状: {batch.x.shape}")
        print(f"  边索引形状: {batch.edge_index.shape}")
        print(f"  标签形状: {batch.y.shape}")
        print(f"  批次索引形状: {batch.batch.shape}")
    
    print(f"\n成功迭代了 {batch_count} 个批次，其中 {empty_batch_count} 个空批次。")
    
    if problem_batches:
        print(f"问题批次索引: {problem_batches[:10]}...")
    
    print("\n数据集验证完成！")
    
except Exception as e:
    print(f"验证数据集过程中出错: {str(e)}")