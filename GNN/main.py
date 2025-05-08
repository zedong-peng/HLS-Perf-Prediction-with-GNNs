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

def train_batch(model, batch, optimizer, task_type, scaler=None):
    """Process training logic for a single batch, with mixed precision support."""
    # Check for shape mismatch
    pred = model(batch)
    if pred.shape[0] != batch.y.shape[0]:
        print(f"Warning: Skipping batch with shape mismatch. Pred shape: {pred.shape}, Label shape: {batch.y.shape}")
        return None
    
    # Calculate loss
    is_labeled = batch.y == batch.y
    pred = pred.to(torch.float32)[is_labeled]
    target = batch.y.to(torch.float32)[is_labeled]
    
    if "regression" in task_type:
        loss = reg_criterion(pred, target)
    elif "classification" in task_type:
        loss = cls_criterion(pred, target)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    # Backpropagation
    optimizer.zero_grad()
    if scaler is not None:
        with autocast():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    else:
        loss.backward()
        optimizer.step()
    
    return loss.item()

def eval_batch(model, batch, evaluator, y_true, y_pred):
    """
    Process evaluation logic for a single batch.
    """
    pred = model(batch)
    if pred.shape[0] != batch.y.shape[0]:
        print(f"Warning: Skipping mismatched shapes during evaluation. Pred shape: {pred.shape}, Label shape: {batch.y.shape}")
        return
    y_true.append(batch.y.view(pred.shape).detach().cpu())
    y_pred.append(pred.detach().cpu())

def train(model, device, loader, optimizer, task_type, scaler=None):
    model.train()
    total_loss = 0
    batch_count = 0
    loss_log_interval = 10  # Log average loss every N batches
    
    for step, batch in enumerate(tqdm(loader, desc="Training")):
        if not hasattr(batch, 'x') or not hasattr(batch, 'batch') or batch.x.shape[0] == 0 or batch.batch.shape[0] == 0:
            print(f"Warning: Skipping empty batch. Batch {step}")
            continue
        batch = batch.to(device)
        if batch.x.shape[0] == 1 or (batch.batch.shape[0] > 0 and batch.batch[-1] == 0):
            pass
        else:
            loss = train_batch(model, batch, optimizer, task_type, scaler)
            if loss is not None:
                total_loss += loss
                batch_count += 1
                
                # Print average loss periodically
                if (step + 1) % loss_log_interval == 0:
                    avg_loss = total_loss / batch_count
                    print(f"Batch {step+1}/{len(loader)}, Average Loss: {avg_loss:.6f}")
    
    # Output overall training loss
    if batch_count > 0:
        epoch_loss = total_loss / batch_count
        print(f"Training completed. Epoch average loss: {epoch_loss:.6f}")
        return epoch_loss
    return float('inf')

def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        if not hasattr(batch, 'x') or not hasattr(batch, 'batch') or batch.x.shape[0] == 0 or batch.batch.shape[0] == 0:
            print(f"Warning: Skipping empty batch during evaluation. Batch {step}")
            continue
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                eval_batch(model, batch, evaluator, y_true, y_pred)
    if len(y_true) == 0 or len(y_pred) == 0:
        print("Warning: No valid prediction results. Please check dataset and model.")
        return {"rmse": float("inf")}, np.array([]), np.array([])
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return evaluator.eval(input_dict), y_true, y_pred

def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def monitor_memory(threshold_gb=70):
    """
    Monitor GPU memory usage and take action when approaching threshold
    """
    if torch.cuda.is_available():
        current_memory = torch.cuda.memory_allocated() / (1024**3)
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if current_memory > threshold_gb:
            print(f"Warning: GPU memory usage reached {current_memory:.2f}GB, approaching limit!")
            torch.cuda.empty_cache()
            if torch.cuda.memory_allocated() / (1024**3) > threshold_gb:
                raise MemoryError("Insufficient GPU memory, actively interrupting to prevent OOM")

def main():
    parser = argparse.ArgumentParser(description='GNN model')
    parser.add_argument('--gnn', type=str, default="pna", help='gnn model to use (default: pna)')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--drop_ratio', type=float, default=0.5, help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5, help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300, help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training and validation')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=300, help='maximum number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="dfg_dsp_binary", help='dataset name (default: lut)')
    parser.add_argument('--feature', type=str, default="full", help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="", help='filename to output result (default: )')
    parser.add_argument('--able_amp', action='store_true', help='Enable Automatic Mixed Precision (AMP) training')
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    scaler = GradScaler() if args.able_amp and torch.cuda.is_available() else None
    dataset = PygGraphPropPredDataset(name=args.dataset)
    if args.feature == 'simple':
        print('Using simple feature mode (only top 2 features)')
        dataset.data.x = dataset.data.x[:, :2]
        dataset.data.edge_attr = dataset.data.edge_attr[:, :2]
    else:
        print('Using full feature mode')
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(args.dataset)
    pin_memory = torch.cuda.is_available()
    num_workers = args.num_workers if args.num_workers > 0 else max(2, os.cpu_count() // 2)
    if args.num_workers > 0:
        print(f"Using specified num_workers={num_workers}")
    else:
        print(f"Automatically set num_workers={num_workers} (half of CPU cores)")
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    standard_test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    model_args = {'num_tasks': dataset.num_tasks, 'num_layer': args.num_layer, 'emb_dim': args.emb_dim, 'drop_ratio': args.drop_ratio}
    if args.gnn == "pna":
        max_degree = max(degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long).max().item() for data in dataset[split_idx["train"]])
        deg = torch.zeros(max_degree + 1, dtype=torch.long)
        for data in dataset[split_idx["train"]]:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())
        model_args['deg'] = deg

    if args.gnn == "pna":
        from pna import Net
    elif args.gnn == "rgcn":
        from rgcn import Net
    elif args.gnn == "gat":
        from gat import Net
        model_args['heads'] = 8
    elif args.gnn == "sage":
        from sage import Net
    elif args.gnn == "arma":
        from ARMA import Net
    elif args.gnn == "film":
        from film import Net
    elif args.gnn == "ggnn":
        from ggnn import Net
    elif args.gnn == "pan":
        from pan import Net
    elif args.gnn == "sgn":
        from sgn import Net
    elif args.gnn == "unet":
        from unet import Net
    elif args.gnn in ['gin', 'gin-virtual', 'gcn', 'gcn-virtual']:
        from gnn import GNN as Net
        model_args['gnn_type'] = args.gnn.split('-')[0]
        model_args['virtual_node'] = '-virtual' in args.gnn
    else:
        raise ValueError(f'--gnn value not support: {args.gnn}. Available options: pna, rgcn, gat, sage, arma, film, ggnn, pan, sgn, unet, gin-virtual, gcn-virtual, gin, gcn')

    model = Net(**model_args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10, min_lr=0.00001, verbose=True)

    valid_curve = []
    test_curve = []
    train_curve = []
    test_predict_value = []
    test_true_value = []
    valid_predict_value = []
    valid_true_value = []

    # Add training loss curve record
    train_loss_curve = []

    if args.able_amp and torch.cuda.is_available():
        print("Automatic Mixed Precision (AMP) training enabled")
    else:
        if not args.able_amp:
            print("Mixed precision training disabled")
        elif not torch.cuda.is_available():
            print("Cannot use mixed precision training: CUDA not available")

    for epoch in range(1, args.epochs + 1):
        print(f"===== Epoch {epoch}/{args.epochs} =====")
        print('Training phase...')
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current learning rate: {current_lr:.6f}')
        try:
            # Replace original training loop with improved train function, record loss
            epoch_loss = train(model, device, train_loader, optimizer, dataset.task_type, scaler)
            train_loss_curve.append(epoch_loss)

            print('Evaluation phase...')
            train_perf, train_true, train_pred = eval(model, device, train_loader, evaluator)
            valid_perf, v_true, v_pred = eval(model, device, valid_loader, evaluator)
            test_perf, t_true, t_pred = eval(model, device, standard_test_loader, evaluator)

            if 'classification' not in dataset.task_type:
                train_true_list = reduce(operator.add, train_true.tolist())
                train_pred_list = reduce(operator.add, train_pred.tolist())
                train_mape = calculate_mape(train_true_list, train_pred_list)
                train_perf['mape'] = train_mape

                valid_true_list = reduce(operator.add, v_true.tolist())
                valid_pred_list = reduce(operator.add, v_pred.tolist())
                valid_mape = calculate_mape(valid_true_list, valid_pred_list)
                valid_perf['mape'] = valid_mape

                test_true_list = reduce(operator.add, t_true.tolist())
                test_pred_list = reduce(operator.add, t_pred.tolist())
                test_mape = calculate_mape(test_true_list, test_pred_list)
                test_perf['mape'] = test_mape

                print({'Train': {'rmse': train_perf[dataset.eval_metric], 'mape': f'{train_mape:.2f}%'},
                       'Validation': {'rmse': valid_perf[dataset.eval_metric], 'mape': f'{valid_mape:.2f}%'},
                       'Test': {'rmse': test_perf[dataset.eval_metric], 'mape': f'{test_mape:.2f}%'}})
            else:
                print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

            train_curve.append(train_perf[dataset.eval_metric])
            valid_curve.append(valid_perf[dataset.eval_metric])
            test_curve.append(test_perf[dataset.eval_metric])

            test_predict_value.append(reduce(operator.add, t_pred.tolist()))
            valid_predict_value.append(reduce(operator.add, v_pred.tolist()))

            test_loss = test_perf[dataset.eval_metric]
            if test_loss <= np.min(np.array(test_curve)):
                PATH = 'model/' + args.dataset + f'_{args.gnn}_layer_' + str(args.num_layer) + '_model.pt'
                os.makedirs(os.path.dirname(PATH), exist_ok=True)
                print(f"Saving best model checkpoint to {PATH}")
                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': test_loss
                            }, PATH)

            test_true_value = reduce(operator.add, t_true.tolist())
            valid_true_value = reduce(operator.add, v_true.tolist())

            if 'classification' in dataset.task_type:
                best_val_epoch = np.argmax(np.array(valid_curve))
                best_train = max(train_curve)
            else:
                best_val_epoch = np.argmin(np.array(valid_curve))
                best_train = min(train_curve)

            print('Epoch training completed!')
            print(f'Best validation score (epoch {best_val_epoch+1}): {valid_curve[best_val_epoch]:.6f}')
            print(f'Test score at best validation epoch: {test_curve[best_val_epoch]:.6f}')

            os.makedirs('result', exist_ok=True)
            f = open('result/' + args.dataset + f'_{args.gnn}_layer_' + str(args.num_layer) + '_training.json', 'w')

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
                    'test': test_curve,
                    'train_loss': train_loss_curve  # Add training loss curve
                }
            }

            if 'classification' not in dataset.task_type:
                final_test_mape = calculate_mape(test_true_value, test_predict_value[-1])
                result['metrics']['test_mape'] = final_test_mape
                print(f'Final test MAPE: {final_test_mape:.2f}%')

                mape_curve = []
                for epoch_preds in test_predict_value:
                    epoch_mape = calculate_mape(test_true_value, epoch_preds)
                    mape_curve.append(epoch_mape)
                result['curves']['mape'] = mape_curve

                best_preds = test_predict_value[best_val_epoch]
                errors = [abs(pred - true) for pred, true in zip(best_preds, test_true_value)]
                result['error_stats'] = {
                    'min': min(errors),
                    'max': max(errors),
                    'mean': sum(errors) / len(errors),
                    'median': sorted(errors)[len(errors) // 2],
                    'percentile_90': sorted(errors)[int(len(errors) * 0.9)]
                }

            json.dump(result, f, indent=4)
            f.close()

            print(f"Results saved to: result/{args.dataset}_{args.gnn}_layer_{args.num_layer}_training.json")

            if not args.filename == '':
                torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch], 'Train': train_curve[best_val_epoch], 'BestTrain': best_train}, args.filename)

            scheduler.step(valid_perf[dataset.eval_metric])

        except MemoryError as e:
            print("Memory pressure detected, safely saving model and adjusting parameters...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'emergency_checkpoint_epoch_{epoch}_batch_{batch_idx}.pt')

            new_batch_size = max(1, args.batch_size // 2)
            print(f"Reducing batch size from {args.batch_size} to {new_batch_size}")
            args.batch_size = new_batch_size

            train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True)
            print("Continuing training with reduced batch size...")

if __name__ == "__main__":
    main()
