import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from pna import Net
from torch_geometric.utils import degree
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

def train(model, device, loader, optimizer, task_type):
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
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

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

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PNA model')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train (default: 300)')
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
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

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

    train_dataset = dataset[split_idx["train"]]
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    
    # 加载标准测试集
    standard_test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    
    # 加载外部测试集（如果提供, 那么覆盖标准测试集）
    external_test_loader = None
    if args.external_test:
        print(f"Loading external test dataset from {args.external_test}")
        try:
            external_dataset = PygGraphPropPredDataset(name=args.external_test, root="./")
            if args.feature == 'simple':
                external_dataset.data.x = external_dataset.data.x[:,:2]
                external_dataset.data.edge_attr = external_dataset.data.edge_attr[:,:2]
            external_test_loader = DataLoader(external_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            print(f"External test dataset loaded with {len(external_dataset)} samples")
        except Exception as e:
            print(f"Failed to load external test dataset: {e}")
            external_test_loader = None

    # Compute in-degree histogram over training data.
    deg = torch.zeros(80, dtype=torch.long)
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

    model = Net(deg=deg, num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10,min_lr=0.00001)

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
        train(model, device, train_loader, optimizer, dataset.task_type)

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
            PATH='model/'+args.dataset + '_pna_layer_'+ str(args.num_layer)+'_model.pt'
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
        best_model = Net(deg=deg, num_tasks=dataset.num_tasks, num_layer=args.num_layer, emb_dim=args.emb_dim, drop_ratio=args.drop_ratio).to(device)
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
        ext_result_file = 'result/'+args.dataset+'_external_test_pna_layer_'+str(args.num_layer)+'.json'
        with open(ext_result_file, 'w') as f:
            json.dump(external_result, f)
        print(f'External test results saved to {ext_result_file}')

    # 保存标准测试结果
    os.makedirs('result', exist_ok=True)
    f = open('result/'+args.dataset + '_pna_layer_'+str(args.num_layer)+'.json', 'w')
    result=dict(val=valid_curve[best_val_epoch], \
        test=test_curve[best_val_epoch],train=train_curve[best_val_epoch], \
        test_pred=test_predict_value, value_pred=valid_predict_value, 
        test_true=test_true_value, valid_true=valid_true_value,\
        train_curve=train_curve, test_curve=test_curve, valid_curve=valid_curve)
    
    # 添加MAPE到结果（如果是回归任务）
    if 'classification' not in dataset.task_type:
        # 计算最后一个epoch的MAPE
        final_test_mape = calculate_mape(test_true_value, test_predict_value[-1])
        result['test_mape'] = final_test_mape
        print(f'Final test MAPE: {final_test_mape:.2f}%')
    
    json.dump(result, f)
    f.close()

    if not args.filename == '':
        torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch], 'Train': train_curve[best_val_epoch], 'BestTrain': best_train}, args.filename)


if __name__ == "__main__":
    main()
