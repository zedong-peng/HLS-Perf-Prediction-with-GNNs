#!/usr/bin/env python3
"""
模型评估脚本
============

加载训练好的模型，对指定数据集进行评估。

用法:
    python eval.py --model_path <模型路径> --ood_design_base_dir <评估数据目录> [其他参数]

注意：默认使用验证集上的最佳模型（best model），这是学术论文的标准做法。

Author: Zedong Peng
"""

import torch
from torch_geometric.data import DataLoader
import os
import json
import argparse
import matplotlib.pyplot as plt
import sys
from typing import Any

# 直接导入训练脚本中的所有必要组件，避免重复代码
try:
    from delta_e2e.train_e2e import (
        E2EDifferentialProcessor, SimpleDifferentialGNN, 
        E2EDifferentialDataset, evaluate_model, differential_collate_fn,
        resolve_loss_function, METRIC_INDEX, _extract_metric_values, _compute_basic_stats, canonical_metric_name
    )
except ImportError:
    # 如果作为模块导入失败，使用相对导入
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_e2e import (
        E2EDifferentialProcessor, SimpleDifferentialGNN,
        E2EDifferentialDataset, evaluate_model, differential_collate_fn,
        resolve_loss_function, METRIC_INDEX, _extract_metric_values, _compute_basic_stats, canonical_metric_name
    )


def create_prediction_plots_eval(metrics: dict, target_metric: str, output_dir: str):
    """创建设计指标预测vs真实值散点图（评估版本，不依赖SwanLab）"""
    design_true = metrics['design_true'].cpu().numpy()
    design_preds = metrics['design_preds'].cpu().numpy()

    plt.figure(figsize=(8, 6))
    plt.scatter(design_true, design_preds, alpha=0.6, s=50)
    min_val = min(design_true.min(), design_preds.min())
    max_val = max(design_true.max(), design_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='Perfect Prediction')
    plt.xlabel(f'True {target_metric.upper()} (Real Resource Units)')
    plt.ylabel(f'Predicted {target_metric.upper()} (Real Resource Units)')
    plt.title(f'Design Performance Prediction ({target_metric.upper()}) - Real Resource Scale')
    plt.legend()
    plt.grid(True, alpha=0.3)

    scatter_path = os.path.join(output_dir, f'eval_prediction_scatter_{target_metric}.png')
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    return scatter_path


def parse_bool_flag(value, default=None):
    """将字符串标志转换为布尔值，保留原始布尔类型。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {'true', '1', 'yes', 'y', 'on'}:
            return True
        if lowered in {'false', '0', 'no', 'n', 'off'}:
            return False
    return default if default is not None else value


def load_model_checkpoint(model_path: str, device: torch.device):
    """加载训练好的模型checkpoint并提取配置"""
    print(f"加载模型: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    args_dict = checkpoint.get('args', {})
    
    # 提取模型配置
    model_config = {
        'target_metric': args_dict.get('target_metric', 'dsp'),
        'hidden_dim': args_dict.get('hidden_dim', 128),
        'num_layers': args_dict.get('num_layers', 3),
        'dropout': args_dict.get('dropout', 0.1),
        'gnn_type': args_dict.get('gnn_type', 'gcn'),
        'differential': parse_bool_flag(args_dict.get('differential', True), True),
        'kernel_baseline': args_dict.get('kernel_baseline', 'learned'),
        'node_dim': None  # 将在加载数据后确定
    }
    
    return checkpoint, args_dict, model_config


def to_serializable(value: Any):
    """递归地将结果转换为可JSON序列化的Python原生类型。"""
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    return value


def load_normalizers_from_stats(model_path: str):
    """从模型目录的 target_metric_stats.json 中加载 median/scale 正则化配置。"""
    model_dir = os.path.dirname(os.path.abspath(model_path))
    stats_path = os.path.join(model_dir, "target_metric_stats.json")
    if not os.path.exists(stats_path):
        return None
    try:
        stats = json.load(open(stats_path, 'r'))
        norm = stats.get("normalizer", {})
        if not norm:
            return None
        return norm
    except Exception:
        return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='评估训练好的差分学习GNN模型')
    
    # 必需参数
    parser.add_argument('--model_path', type=str, required=True,
                        help='训练好的模型路径 (.pt文件)')
    parser.add_argument('--ood_design_base_dir', type=str, required=True,
                        help='评估数据集根目录（design数据）')
    
    # 可选参数（如果模型文件中没有保存）
    parser.add_argument('--kernel_base_dir', type=str,
                        default='/home/user/zedongpeng/workspace/HLS-Perf-Prediction-with-GNNs/Graphs/forgehls_kernels/',
                        help='Kernel数据根目录（用于配对）')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='评估结果输出目录（默认：模型所在目录/eval_results）')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--device', type=int, default=0, help='GPU设备ID')
    parser.add_argument('--loader_workers', type=int, default=8, help='DataLoader worker数量')
    parser.add_argument('--cache_root', type=str, default="./graph_cache",
                        help='图数据缓存根目录（默认：模型所在目录的graph_cache）')
    parser.add_argument('--rebuild_cache', action='store_true',
                        help='忽略已有缓存并重新构建')
    parser.add_argument('--max_workers', type=int, default=-1,
                        help='数据处理并行线程数（<=0表示自动）')
    parser.add_argument('--loss_type', type=str, default=None,
                        choices=['mae', 'mse', 'smoothl1'],
                        help='评估时使用的损失函数（默认使用模型训练时的配置）')
    # 代码特征相关（默认使用模型训练时的配置，可选覆盖）
    parser.add_argument('--use_code_feature', type=str, default=None, choices=['true', 'false'],
                        help='是否启用设计源码嵌入特征（默认使用模型保存的配置）')
    parser.add_argument('--code_model_path', type=str, default=None,
                        help='代码特征模型路径（启用代码特征时必填；默认使用模型保存的路径）')
    parser.add_argument('--code_cache_root', type=str, default=None,
                        help='代码嵌入缓存根目录（默认跟随 cache_root）')
    parser.add_argument('--code_pooling', type=str, default=None, choices=['last_token', 'mean'],
                        help='代码特征 pooling 策略')
    parser.add_argument('--code_max_length', type=int, default=None, help='代码序列最大长度')
    parser.add_argument('--code_normalize', type=str, default=None, choices=['true', 'false'],
                        help='是否对代码特征做 L2 归一化')
    parser.add_argument('--code_batch_size', type=int, default=None, help='预编码代码特征批大小')
    parser.add_argument('--graph_pooling', type=str, default=None,
                        choices=['sum', 'mean', 'max', 'attention', 'set2set'],
                        help='图池化方式（默认使用模型训练时的配置）')
    parser.add_argument('--apply_hard_filter', type=str, default=None, choices=['true', 'false'],
                        help='是否对评估集做 p05-p95 硬过滤（默认沿用模型训练配置）')
    parser.add_argument('--use_normalizer', type=str, default='true', choices=['true', 'false'],
                        help='是否从 target_metric_stats.json 加载 normalizer 用于评估')
    parser.add_argument('--filter_resource_mismatch', type=str, default=None, choices=['true', 'false'],
                        help='当 adb 细分 DSP 总和与 csynth DSP 不一致时是否过滤样本（默认沿用模型训练配置）')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型checkpoint
    checkpoint, saved_args, model_config = load_model_checkpoint(args.model_path, device)
    
    # 从保存的模型配置中自动加载所有超参数
    kernel_base_dir = saved_args.get('kernel_base_dir', args.kernel_base_dir)
    loss_type = (args.loss_type if args.loss_type is not None else saved_args.get('loss_type', 'mae'))
    loss_type = str(loss_type).lower()
    loss_fn = resolve_loss_function(loss_type)
    args.loss_type = loss_type
    
    # 自动从模型配置加载hierarchical和region（训练时保存的配置）
    # 不设置默认值；若缺失直接报错
    if 'hierarchical' not in saved_args:
        raise ValueError("评估失败：模型checkpoint未保存 'hierarchical' 配置，请使用包含完整args的训练脚本重新训练并保存模型。")
    if 'region' not in saved_args:
        raise ValueError("评估失败：模型checkpoint未保存 'region' 配置，请使用包含完整args的训练脚本重新训练并保存模型。")
    hierarchical_from_model = saved_args.get('hierarchical')
    region_from_model = saved_args.get('region')
    
    # 统一转换为字符串格式
    if isinstance(hierarchical_from_model, bool):
        hierarchical_flag = 'on' if hierarchical_from_model else 'off'
    elif isinstance(hierarchical_from_model, str):
        hierarchical_flag = hierarchical_from_model.lower()
    else:
        raise ValueError("评估失败：无法解析 'hierarchical' 配置类型，期望为 bool 或 str(on/off)。")
    
    if isinstance(region_from_model, bool):
        region_flag = 'on' if region_from_model else 'off'
    elif isinstance(region_from_model, str):
        region_flag = region_from_model.lower()
    else:
        raise ValueError("评估失败：无法解析 'region' 配置类型，期望为 bool 或 str(on/off)。")
    
    # 代码特征 / 池化等模型配置（使用训练时配置，可被CLI覆盖）
    use_code_feature = parse_bool_flag(args.use_code_feature, parse_bool_flag(saved_args.get('use_code_feature', False), False))
    code_model_path = args.code_model_path or saved_args.get('code_model_path')
    code_cache_root = args.code_cache_root or args.cache_root
    code_pooling = args.code_pooling or saved_args.get('code_pooling', 'last_token')
    code_max_length = args.code_max_length if args.code_max_length is not None else saved_args.get('code_max_length', 2048)
    code_normalize = parse_bool_flag(args.code_normalize, parse_bool_flag(saved_args.get('code_normalize', False), False))
    code_batch_size = args.code_batch_size if args.code_batch_size is not None else saved_args.get('code_batch_size', 8)
    graph_pooling = args.graph_pooling or saved_args.get('graph_pooling', 'sum')
    kernel_baseline = saved_args.get('kernel_baseline', 'learned')
    differential_mode = parse_bool_flag(saved_args.get('differential', True), True)

    if use_code_feature and not code_model_path:
        raise ValueError("评估失败：模型开启了 use_code_feature，但未提供 code_model_path。请通过 --code_model_path 指定。")
    
    # 记录解析后的配置，便于后续保存
    args.use_code_feature = use_code_feature
    args.code_model_path = code_model_path
    args.code_cache_root = code_cache_root
    args.code_pooling = code_pooling
    args.code_max_length = code_max_length
    args.code_normalize = code_normalize
    args.code_batch_size = code_batch_size
    args.graph_pooling = graph_pooling

    print(f"自动加载模型配置: hierarchical={hierarchical_flag}, region={region_flag}")
    target_metric = canonical_metric_name(model_config['target_metric'])
    print(f"评估目标指标: {target_metric.upper()}（与训练配置保持一致）")
    print(f"评估损失函数: {loss_type}")
    
    # 如果没有提供output_dir，使用模型所在目录
    if args.output_dir is None:
        model_dir = os.path.dirname(os.path.abspath(args.model_path))
        output_dir = os.path.join(model_dir, 'eval_results')
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"结果将保存到: {output_dir}")
    
    # 确定缓存目录
    if args.cache_root is None:
        model_dir = os.path.dirname(os.path.abspath(args.model_path))
        cache_root = os.path.join(model_dir, 'graph_cache')
    else:
        cache_root = args.cache_root
    
    # ==================== 加载评估数据 ====================
    print("\n加载评估数据...")
    processor = E2EDifferentialProcessor(
        kernel_base_dir=kernel_base_dir,
        design_base_dir=args.ood_design_base_dir,
        output_dir=output_dir,
        cache_root=cache_root,
        rebuild_cache=args.rebuild_cache,
        hierarchical=(hierarchical_flag == 'on'),
        region=(region_flag == 'on'),
        max_workers=(None if args.max_workers <= 0 else args.max_workers),
        use_code_feature=use_code_feature,
        code_model_path=code_model_path,
        code_cache_root=code_cache_root,
        code_pooling=code_pooling,
        code_max_length=code_max_length,
        code_normalize=code_normalize,
        code_batch_size=code_batch_size,
        graph_pooling=graph_pooling,
        filter_resource_mismatch=filter_resource_mismatch
    )
    
    eval_pairs = processor.collect_all_data()
    # 若流水线异常返回空，但缓存存在，则尝试直接从缓存加载
    if not eval_pairs:
        eval_pairs = processor._load_cached_pairs(materialize=True)
    if use_code_feature and eval_pairs:
        eval_pairs = processor.attach_code_features(eval_pairs)

    apply_hard_filter = parse_bool_flag(args.apply_hard_filter, parse_bool_flag(saved_args.get('apply_hard_filter', True), True))
    filter_resource_mismatch = parse_bool_flag(args.filter_resource_mismatch, parse_bool_flag(saved_args.get('filter_resource_mismatch', False), False))
    if apply_hard_filter and eval_pairs:
        thresholds = {}
        try:
            metric_idx = METRIC_INDEX[target_metric]
            delta_key = f"{target_metric}_delta"
            stats = _extract_metric_values(eval_pairs, metric_idx, delta_key)
            stats_basic = {k: _compute_basic_stats(vals) for k, vals in stats.items()}
            thresholds = {k: (v["p05"], v["p95"]) for k, v in stats_basic.items() if v}
        except Exception as exc:
            print(f"[Eval] 计算硬过滤阈值失败，将跳过过滤: {exc}")
    
        if thresholds:
            filtered = []
            for pair in eval_pairs:
                vals = _extract_metric_values([pair], metric_idx, delta_key)
                keep = True
                for key, (low, high) in thresholds.items():
                    value = vals[key][0]
                    if value < low or value > high:
                        keep = False
                        break
                if keep:
                    filtered.append(pair)
            dropped = len(eval_pairs) - len(filtered)
            eval_pairs = filtered
            print(f"[Eval] 硬过滤(p05-p95) 移除 {dropped} 条，剩余 {len(eval_pairs)} 条")
    
    if not eval_pairs:
        print("错误: 未找到有效的评估数据配对")
        return
    
    print(f"找到 {len(eval_pairs)} 个评估配对")
    
    # 获取一个样本以确定节点维度
    sample_pair = eval_pairs[0]
    node_dim = sample_pair['kernel_graph'].x.size(1)
    code_dim = None
    if use_code_feature:
        sample_code = sample_pair.get('design_code_embedding')
        if sample_code is None:
            raise ValueError("评估失败：use_code_feature=True 但样本缺少 design_code，请检查代码特征缓存或提供正确的 code_model_path。")
        code_dim = sample_code.shape[-1]
    
    # 如果模型配置中没有node_dim，更新它
    if model_config['node_dim'] is None:
        model_config['node_dim'] = node_dim
    model_config.update({
        'graph_pooling': graph_pooling,
        'kernel_baseline': kernel_baseline,
        'differential': differential_mode,
        'use_code_feature': use_code_feature,
        'code_dim': code_dim
    })
    
    # ==================== 创建模型并加载权重 ====================
    print("\n创建模型...")
    model = SimpleDifferentialGNN(
        node_dim=node_dim,
        hidden_dim=model_config['hidden_dim'],
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout'],
        target_metric=model_config['target_metric'],
        differential=model_config['differential'],
        gnn_type=model_config['gnn_type'],
        kernel_baseline=model_config['kernel_baseline'],
        use_code_feature=use_code_feature,
        code_dim=code_dim,
        graph_pooling=graph_pooling
    ).to(device)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("模型权重加载完成")
    normalizers = None
    if args.use_normalizer.lower() == 'true':
        normalizers = load_normalizers_from_stats(args.model_path)
    
    # ==================== 创建数据集和数据加载器 ====================
    dataset = E2EDifferentialDataset(eval_pairs, model_config['target_metric'])
    
    loader_workers = max(0, int(args.loader_workers))
    loader_kwargs = {
        'collate_fn': differential_collate_fn,
        'num_workers': loader_workers,
        'pin_memory': True if loader_workers > 0 else False,
        'persistent_workers': (loader_workers > 0)
    }
    
    if loader_workers > 0:
        try:
            loader_kwargs['multiprocessing_context'] = 'spawn'
        except Exception:
            pass
    
    eval_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        **loader_kwargs
    )
    
    # ==================== 评估模型 ====================
    print("\n开始评估...")
    metrics = evaluate_model(
        model, eval_loader, device, 
        target_metric=model_config['target_metric'],
        return_predictions=False,
        loss_fn=loss_fn,
        normalizers=normalizers
    )
    
    # ==================== 保存结果 ====================
    print("\n评估结果:")
    print(f"  设计MAE({target_metric.upper()}): {metrics.get('design_mae', 0):.6f}")
    print(f"  设计RMSE({target_metric.upper()}): {metrics.get('design_rmse', 0):.6f}")
    print(f"  设计MAPE({target_metric.upper()}): {metrics.get('design_mape', 0):.2f}%")
    print(f"  设计R²({target_metric.upper()}): {metrics.get('design_r2', 0):.4f}")
    if 'design_ulti_rmse' in metrics:
        print(f"  设计ulti-RMSE: {metrics.get('design_ulti_rmse', 0):.8f}")
    
    if 'kernel_mae' in metrics:
        print(f"  Kernel MAE: {metrics.get('kernel_mae', 0):.6f}")
        print(f"  Kernel RMSE: {metrics.get('kernel_rmse', 0):.6f}")
        print(f"  Kernel MAPE: {metrics.get('kernel_mape', 0):.2f}%")
        print(f"  Kernel R²: {metrics.get('kernel_r2', 0):.4f}")
        if 'kernel_ulti_rmse' in metrics:
            print(f"  Kernel ulti-RMSE: {metrics.get('kernel_ulti_rmse', 0):.8f}")
    
    if 'delta_mae' in metrics:
        print(f"  差值MAE: {metrics.get('delta_mae', 0):.6f}")
        print(f"  差值RMSE: {metrics.get('delta_rmse', 0):.6f}")
        print(f"  差值MAPE: {metrics.get('delta_mape', 0):.2f}%")
        print(f"  差值R²: {metrics.get('delta_r2', 0):.4f}")
        if 'delta_ulti_rmse' in metrics:
            print(f"  差值ulti-RMSE: {metrics.get('delta_ulti_rmse', 0):.8f}")
    
    # 保存评估结果到JSON
    eval_results = {
        'model_path': args.model_path,
        'eval_data_dir': args.ood_design_base_dir,
        'num_pairs': len(eval_pairs),
        'target_metric': target_metric,
        'loss_type': loss_type,
        'metrics': to_serializable({k: v for k, v in metrics.items()}),
        'model_config': model_config,
        'eval_config': vars(args)
    }
    
    # 记录训练时保存的最佳结果信息，便于对齐
    checkpoint_training_summary = {
        'best_epoch': checkpoint.get('epoch'),
        'best_valid_loss': checkpoint.get('valid_loss'),
        'id_test_metrics': to_serializable(checkpoint.get('id_test_metrics')),
        'ood_test_metrics': to_serializable(checkpoint.get('ood_test_metrics'))
    }
    eval_results['training_summary'] = checkpoint_training_summary
    
    results_path = os.path.join(output_dir, 'eval_results.json')
    with open(results_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"\n结果已保存到: {results_path}")
    
    # 创建预测散点图
    if 'design_preds' in metrics and 'design_true' in metrics:
        try:
            scatter_path = create_prediction_plots_eval(metrics, model_config['target_metric'], output_dir)
            print(f"预测散点图已保存: {scatter_path}")
        except Exception as e:
            print(f"警告: 无法创建预测散点图: {e}")
    
    print(f"\n评估完成！结果保存在: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n检测到中断，正在退出...")
        sys.exit(130)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
