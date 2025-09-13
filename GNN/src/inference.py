import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from gnn import GNN
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.utils import degree

from tqdm import tqdm
import argparse
import time
import numpy as np
import json
import operator
from functools import reduce
import os

import ARMA
import film
import gat
import pna
import pan
import sage
import sgn
import unet
import rgcn
import ggnn

### importing evaluator
from dataset_pyg import PygGraphPropPredDataset
from evaluate import Evaluator

reg_criterion = torch.nn.MSELoss()
#reg_criterion=torch.nn.SmoothL1Loss(reduction='mean', beta=1.0)


def train(model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss.backward()
            optimizer.step()

def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        # 检查批次是否为空或无效
        if not hasattr(batch, 'x') or not hasattr(batch, 'batch') or batch.x.shape[0] == 0 or batch.batch.shape[0] == 0:
            print(f"警告：评估时跳过空批次。批次 {step}")
            continue
            
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)
                
            # 检查预测值和标签的形状是否匹配
            if pred.shape[0] != batch.y.shape[0]:
                print(f"警告：评估时跳过形状不匹配的批次。预测形状: {pred.shape}, 标签形状: {batch.y.shape}")
                continue

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    # 检查是否有收集到预测结果
    if len(y_true) == 0 or len(y_pred) == 0:
        print("警告：没有有效的预测结果。请检查数据集和模型。")
        return {"rmse": float("inf")}, np.array([]), np.array([])

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict), y_true, y_pred

# 添加MAPE计算函数
def calculate_mape(y_true, y_pred):
    """
    计算平均绝对百分比误差 (MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # 避免除以零
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# 添加RMSE计算函数
def calculate_rmse(y_true, y_pred):
    """
    计算均方根误差 (RMSE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def monitor_memory(threshold_gb=70):
    """监控GPU内存，在接近阈值时采取行动"""
    if torch.cuda.is_available():
        current_memory = torch.cuda.memory_allocated() / (1024**3)
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        if current_memory > threshold_gb:
            print(f"警告: GPU内存使用达到{current_memory:.2f}GB，接近限制！")
            # 主动释放可能不必要的缓存
            torch.cuda.empty_cache()
            # 如果仍然接近阈值，中断当前批次
            if torch.cuda.memory_allocated() / (1024**3) > threshold_gb:
                raise MemoryError("GPU内存不足，主动中断以防止OOM")

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                        help='dataset name (default: ogbg-molhiv)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='path to the .pt model file (default: None, will use auto-generated path)')
    parser.add_argument('--output_dir', type=str, default='./result',
                        help='directory to save inference results (default: ./result)')
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name = args.dataset)

    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = 0)

    if args.gnn == 'gin':
        model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    elif args.gnn == 'arma':
        model = ARMA.Net(num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio).to(device)
    elif args.gnn == 'film':
        model = film.Net(num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio).to(device)
    elif args.gnn == 'sgn':
        model = sgn.Net(num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio).to(device)
    elif args.gnn == 'sage':
        model = sage.Net(num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio).to(device)
    elif args.gnn == 'gat':
        model = gat.Net(heads=8, num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio).to(device)
    elif args.gnn == 'pna':
        deg = torch.zeros(80, dtype=torch.long)
        train_dataset = dataset[split_idx["train"]]
        for data in train_dataset:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())
        model = pna.Net(deg=deg, num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio).to(device)
    elif args.gnn == 'pan':
        model = pan.Net(num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio).to(device)
    elif args.gnn == 'unet':
        model= unet.Net(num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio).to(device)
    elif args.gnn == 'rgcn':
        model = rgcn.Net(num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio).to(device)
    elif args.gnn == 'ggnn':
        model = ggnn.Net(num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio).to(device)
    else:
        raise ValueError('Invalid GNN type')


    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10,min_lr=0.00001)

    
    checkpoint = torch.load(args.model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    test_perf, t_true, t_pred = eval(model, device, test_loader, evaluator)
    
    test_true_value = reduce(operator.add, t_true.tolist())
    test_pred_value = reduce(operator.add, t_pred.tolist())

    # 计算MAPE和RMSE
    mape = calculate_mape(test_true_value, test_pred_value)
    rmse = calculate_rmse(test_true_value, test_pred_value)
    
    # 计算误差统计信息
    errors = [abs(pred - true) for pred, true in zip(test_pred_value, test_true_value)]
    error_stats = {
        'min': float(min(errors)),
        'max': float(max(errors)),
        'mean': float(sum(errors) / len(errors)),
        'median': float(sorted(errors)[len(errors)//2]),
        'percentile_90': float(sorted(errors)[int(len(errors)*0.9)])
    }
    
    # 确保结果目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存结果
    result = {
        'metrics': {
            'test': float(test_perf[dataset.eval_metric]),
            'test_mape': float(mape),
            'test_rmse': float(rmse)
        },
        'error_stats': error_stats,
        'model_info': {
            'model_path': args.model_path,
            'gnn_type': args.gnn,
            'num_layer': args.num_layer,
            'emb_dim': args.emb_dim,
            'drop_ratio': args.drop_ratio
        },
        'dataset_info': {
            'name': args.dataset,
            'num_samples': len(dataset[split_idx["test"]])
        },
        'test_true': test_true_value,
        'test_pred': test_pred_value
    }
    
    # 使用传入的output_dir参数构建输出文件路径
    output_file = os.path.join(args.output_dir, f'inference_results.json')
    
    # 使用indent参数使JSON格式化输出，更易读
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)
    
    print(f"推理完成。测试集RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
    print(f"结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
