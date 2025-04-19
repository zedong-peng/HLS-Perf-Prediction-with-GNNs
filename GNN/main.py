import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.utils import degree
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch_geometric.transforms as T
from torch.cuda.amp import GradScaler, autocast

from tqdm import tqdm
import argparse
import time
import numpy as np
import json
import operator
from functools import reduce
import os

from dataset_pyg import PygGraphPropPredDataset
from evaluate import Evaluator

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

def train(model, device, loader, optimizer, task_type, scaler=None):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        # 检查批次是否为空或无效
        if not hasattr(batch, 'x') or not hasattr(batch, 'batch') or batch.x.shape[0] == 0 or batch.batch.shape[0] == 0:
            print(f"警告：跳过空批次。批次 {step}")
            continue
            
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or (batch.batch.shape[0] > 0 and batch.batch[-1] == 0):
            pass
        else:
            # 使用混合精度训练
            if scaler is not None:
                with autocast():
                    pred = model(batch)
                    ## ignore nan targets (unlabeled) when computing training loss.
                    is_labeled = batch.y == batch.y
                    
                    # 检查预测值和标签的形状是否匹配
                    if pred.shape[0] != batch.y.shape[0]:
                        print(f"警告：跳过形状不匹配的批次。预测形状: {pred.shape}, 标签形状: {batch.y.shape}")
                        continue
                        
                    if "classification" in task_type: 
                        loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
                    else:
                        loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # 原始训练流程
                pred = model(batch)
                optimizer.zero_grad()
                ## ignore nan targets (unlabeled) when computing training loss.
                is_labeled = batch.y == batch.y
                
                # 检查预测值和标签的形状是否匹配
                if pred.shape[0] != batch.y.shape[0]:
                    print(f"警告：跳过形状不匹配的批次。预测形状: {pred.shape}, 标签形状: {batch.y.shape}")
                    continue
                    
                if "classification" in task_type: 
                    loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
                else:
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
    parser = argparse.ArgumentParser(description='GNN model')
    parser.add_argument('--gnn', type=str, default="pna",
                        help='gnn model to use (default: pna)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for training and validation')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='maximum number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="dfg_dsp_binary",
                        help='dataset name (default: lut)')
    parser.add_argument('--external_test', type=str, default="",
                        help='path to external test dataset (default: "")')
    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    parser.add_argument('--able_amp', action='store_true',
                        help='混合精度训练加速')

    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    # 默认关闭混合精度训练
    scaler = None
    if args.able_amp and torch.cuda.is_available():
        print("启用自动混合精度训练")
        scaler = GradScaler()
    else:
        if not args.able_amp:
            print("混合精度训练已禁用")
        elif not torch.cuda.is_available():
            print("无法使用混合精度训练：CUDA不可用")
    
    ### automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name = args.dataset)

    if args.feature == 'full':
        pass 
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:,:2]
        dataset.data.edge_attr = dataset.data.edge_attr[:,:2]

    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    # 使用pin_memory加速CPU到GPU的数据传输
    pin_memory = torch.cuda.is_available()

    # 自动设置最佳num_workers数量
    if args.num_workers <= 0:
        # 如果用户没有指定或指定为0，则自动设置为CPU核心数的一半（最少为2）
        import multiprocessing
        optimal_workers = max(2, multiprocessing.cpu_count() // 2)
        print(f"自动设置num_workers={optimal_workers} (CPU核心数的一半)")
        num_workers = optimal_workers
    else:
        num_workers = args.num_workers
        print(f"使用指定的num_workers={num_workers}")
    
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, 
                             shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, 
                             shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
    standard_test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, 
                                     shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
    
    # 加载外部测试集（如果提供, 那么覆盖标准测试集）
    external_test_loader = None
    if args.external_test:
        print(f"Loading external test dataset from {args.external_test}")
        try:
            external_dataset = PygGraphPropPredDataset(name=args.external_test, root="./")
            if args.feature == 'simple':
                external_dataset.data.x = external_dataset.data.x[:,:2]
                external_dataset.data.edge_attr = external_dataset.data.edge_attr[:,:2]
            external_test_loader = loader.DataLoader(external_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            print(f"External test dataset loaded with {len(external_dataset)} samples")
        except Exception as e:
            print(f"Failed to load external test dataset: {e}")
            external_test_loader = None

    # 根据gnn选择模型
    if args.gnn == "pna":
        # Compute in-degree histogram over training data.
        # 首先找到训练集中的最大节点度数
        max_degree = 0
        for data in dataset[split_idx["train"]]:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            max_degree = max(max_degree, d.max().item())
        
        # 使用找到的最大度数加1来创建直方图向量
        deg = torch.zeros(max_degree + 1, dtype=torch.long)
        for data in dataset[split_idx["train"]]:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())
        
        from pna import Net
        model = Net(deg=deg, num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.gnn == "rgcn":
        from rgcn import Net
        model = Net(num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    elif args.gnn == "gat":
        from gat import Net
        head=8
        model = Net(heads=head, num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.gnn == "sage":
        from sage import Net
        model = Net(num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 改进的学习率调度器 - 更灵活的配置
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=10,
        min_lr=0.0001,
        verbose=True  # 添加verbose参数以打印学习率变化
    )

    valid_curve = []
    test_curve = []
    train_curve = []

    test_predict_value= []
    test_true_value= []
    valid_predict_value= []
    valid_true_value= []

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        
        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current learning rate: {current_lr}')
        
        try:
            for batch_idx, batch in enumerate(train_loader):
                # 每100个批次检查一次内存(基本上不检查)
                if batch_idx % 100 == 0:
                    monitor_memory(threshold_gb=70)  # 设置适当的阈值
                
                batch = batch.to(device)

                if batch.x.shape[0] == 1 or (batch.batch.shape[0] > 0 and batch.batch[-1] == 0):
                    pass
                else:
                    # 使用混合精度训练
                    if scaler is not None:
                        with autocast():
                            pred = model(batch)
                            ## ignore nan targets (unlabeled) when computing training loss.
                            is_labeled = batch.y == batch.y
                            
                            # 检查预测值和标签的形状是否匹配
                            if pred.shape[0] != batch.y.shape[0]:
                                print(f"警告：跳过形状不匹配的批次。预测形状: {pred.shape}, 标签形状: {batch.y.shape}")
                                continue
                                
                            if "classification" in dataset.task_type: 
                                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
                            else:
                                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
                        
                        optimizer.zero_grad()
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # 原始训练流程
                        pred = model(batch)
                        optimizer.zero_grad()
                        ## ignore nan targets (unlabeled) when computing training loss.
                        is_labeled = batch.y == batch.y
                        
                        # 检查预测值和标签的形状是否匹配
                        if pred.shape[0] != batch.y.shape[0]:
                            print(f"警告：跳过形状不匹配的批次。预测形状: {pred.shape}, 标签形状: {batch.y.shape}")
                            continue
                            
                        if "classification" in dataset.task_type: 
                            loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
                        else:
                            loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
                        loss.backward()
                        optimizer.step()

            print('Evaluating...')
            train_perf, train_true, train_pred = eval(model, device, train_loader, evaluator)
            valid_perf, v_true, v_pred = eval(model, device, valid_loader, evaluator)
            test_perf, t_true, t_pred = eval(model, device, standard_test_loader, evaluator)

            # 计算并添加MAPE指标（如果是回归任务）
            if 'classification' not in dataset.task_type:
                # 计算训练集MAPE
                train_true_list = reduce(operator.add, train_true.tolist())
                train_pred_list = reduce(operator.add, train_pred.tolist())
                train_mape = calculate_mape(train_true_list, train_pred_list)
                train_perf['mape'] = train_mape
                
                # 计算验证集MAPE
                valid_true_list = reduce(operator.add, v_true.tolist())
                valid_pred_list = reduce(operator.add, v_pred.tolist())
                valid_mape = calculate_mape(valid_true_list, valid_pred_list)
                valid_perf['mape'] = valid_mape
                
                # 计算测试集MAPE
                test_true_list = reduce(operator.add, t_true.tolist())
                test_pred_list = reduce(operator.add, t_pred.tolist())
                test_mape = calculate_mape(test_true_list, test_pred_list)
                test_perf['mape'] = test_mape
                
                # 打印包含MAPE的性能指标（添加百分号以保持一致性）
                print({'Train': {'rmse': train_perf[dataset.eval_metric], 'mape': f'{train_mape:.2f}%'}, 
                       'Validation': {'rmse': valid_perf[dataset.eval_metric], 'mape': f'{valid_mape:.2f}%'}, 
                       'Test': {'rmse': test_perf[dataset.eval_metric], 'mape': f'{test_mape:.2f}%'}})
            else:
                # 对于分类任务，只打印原有指标
                print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

            train_curve.append(train_perf[dataset.eval_metric])
            valid_curve.append(valid_perf[dataset.eval_metric])
            test_curve.append(test_perf[dataset.eval_metric])

            test_predict_value.append(reduce(operator.add, t_pred.tolist()))
            valid_predict_value.append(reduce(operator.add, v_pred.tolist()))

            test_loss=test_perf[dataset.eval_metric]
            if test_loss<=np.min(np.array(test_curve)):
                PATH='model/'+args.dataset + f'_{args.gnn}_layer_'+ str(args.num_layer)+'_model.pt'
                os.makedirs(os.path.dirname(PATH), exist_ok=True)  # Ensure the directory exists
                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': test_loss
                            }, PATH)
            
            test_true_value=reduce(operator.add, t_true.tolist())
            valid_true_value=reduce(operator.add, v_true.tolist())

            if 'classification' in dataset.task_type:
                best_val_epoch = np.argmax(np.array(valid_curve))
                best_train = max(train_curve)
            else:
                best_val_epoch = np.argmin(np.array(valid_curve))
                best_train = min(train_curve)

            print('Finished training!')
            print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
            print('Test score: {}'.format(test_curve[best_val_epoch]))

            # 评估外部测试集（如果提供）
            if external_test_loader:
                print('Evaluating on external test set...')
                # 加载最佳模型
                if args.gnn == "pna":
                    best_model = Net(deg=deg, num_tasks=dataset.num_tasks, num_layer=args.num_layer, emb_dim=args.emb_dim, drop_ratio=args.drop_ratio).to(device)
                elif args.gnn == "rgcn":
                    best_model = Net(num_tasks=dataset.num_tasks, num_layer=args.num_layer, emb_dim=args.emb_dim, drop_ratio=args.drop_ratio).to(device)
                elif args.gnn == "gat":
                    head=8
                    best_model = Net(heads=head, num_tasks=dataset.num_tasks, num_layer=args.num_layer, emb_dim=args.emb_dim, drop_ratio=args.drop_ratio).to(device)
                elif args.gnn == "sage":
                    best_model = Net(num_tasks=dataset.num_tasks, num_layer=args.num_layer, emb_dim=args.emb_dim, drop_ratio=args.drop_ratio).to(device)
                    
                best_model.load_state_dict(torch.load(PATH, weights_only=False)['model_state_dict'])
                
                external_perf, ext_true, ext_pred = eval(best_model, device, external_test_loader, evaluator)
                print('External test score: {}'.format(external_perf[dataset.eval_metric]))
                
                # 计算MAPE
                ext_true_list = reduce(operator.add, ext_true.tolist())
                ext_pred_list = reduce(operator.add, ext_pred.tolist())
                
                # 确保是回归任务再计算MAPE
                if 'classification' not in dataset.task_type:
                    mape = calculate_mape(ext_true_list, ext_pred_list)
                    print(f'External test MAPE: {mape:.2f}%')
                else:
                    mape = None
                    print('MAPE not applicable for classification tasks')
                
                # 保存外部测试结果
                external_result = {
                    'external_test': external_perf[dataset.eval_metric],
                    'external_pred': ext_pred_list,
                    'external_true': ext_true_list
                }
                
                # 添加MAPE到结果，同时保留原有指标
                if mape is not None:
                    external_result['external_mape'] = mape
                
                # 将外部测试结果添加到最终结果中
                os.makedirs('result', exist_ok=True)
                ext_result_file = 'result/'+args.dataset+'_external_test_pna_layer_'+str(args.num_layer)+'_training.json'
                with open(ext_result_file, 'w') as f:
                    json.dump(external_result, f, indent=4)
                print(f'External test results saved to {ext_result_file}')

            # 保存标准测试结果
            os.makedirs('result', exist_ok=True)
            f = open('result/'+args.dataset + f'_{args.gnn}_layer_'+str(args.num_layer)+'_training.json', 'w')
            
            # 优化JSON结果，只保存必要信息
            result = {
                'metrics': {
                    'val': valid_curve[best_val_epoch],
                    'test': test_curve[best_val_epoch],
                    'train': train_curve[best_val_epoch]
                },
                'best_epoch': int(best_val_epoch),
                'curves': {
                    'train': train_curve,
                    'valid': valid_curve,
                    'test': test_curve
                }
            }
            
            # 不保存完整的预测值和真实值，只保存统计信息
            if 'classification' not in dataset.task_type:
                # 计算最后一个epoch的MAPE
                final_test_mape = calculate_mape(test_true_value, test_predict_value[-1])
                result['metrics']['test_mape'] = final_test_mape
                print(f'Final test MAPE: {final_test_mape:.2f}%')
                
                # 添加每个epoch的MAPE指标
                mape_curve = []
                for epoch_preds in test_predict_value:
                    epoch_mape = calculate_mape(test_true_value, epoch_preds)
                    mape_curve.append(epoch_mape)
                result['curves']['mape'] = mape_curve
                
                # 可选：保存预测误差的统计信息而不是完整数据
                best_preds = test_predict_value[best_val_epoch]
                errors = [abs(pred - true) for pred, true in zip(best_preds, test_true_value)]
                result['error_stats'] = {
                    'min': min(errors),
                    'max': max(errors),
                    'mean': sum(errors) / len(errors),
                    'median': sorted(errors)[len(errors)//2],
                    'percentile_90': sorted(errors)[int(len(errors)*0.9)]
                }
            
            # 使用indent参数使JSON格式化输出，更易读
            json.dump(result, f, indent=4)
            f.close()

            if not args.filename == '':
                torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch], 'Train': train_curve[best_val_epoch], 'BestTrain': best_train}, args.filename)

            # 在验证后更新学习率调度器
            scheduler.step(valid_perf[dataset.eval_metric])

        except MemoryError as e:
            print(f"检测到内存压力，正在安全保存模型并调整参数...")
            # 保存检查点
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'emergency_checkpoint_epoch_{epoch}_batch_{batch_idx}.pt')
            
            # 减小批处理大小并继续
            new_batch_size = max(1, args.batch_size // 2)
            print(f"将批处理大小从{args.batch_size}减小到{new_batch_size}")
            args.batch_size = new_batch_size
            
            # 重新创建数据加载器
            train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True)
            # 继续训练...


if __name__ == "__main__":
    main()
