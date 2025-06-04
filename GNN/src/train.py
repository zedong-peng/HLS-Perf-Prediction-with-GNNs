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
from datetime import datetime
import matplotlib.pyplot as plt

from dataset_pyg import PygGraphPropPredDataset
from evaluate import Evaluator

import argparse

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

def train_batch(model, batch, optimizer, task_type, scaler=None):
    """Process training logic for a single batch"""
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
    loss.backward()
    optimizer.step()
    
    # Clear intermediate variables to save memory
    del pred, target, is_labeled
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
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
        
        # # Memory optimization: Skip extremely large batches
        # if hasattr(batch, 'num_nodes') and batch.num_nodes > 3000:
        #     print(f"Warning: Skipping large batch with {batch.num_nodes} nodes to prevent OOM")
        #     continue
            
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
        
        # # Memory optimization: Skip extremely large batches during evaluation too
        # if hasattr(batch, 'num_nodes') and batch.num_nodes > 3000:
        #     print(f"Warning: Skipping large evaluation batch with {batch.num_nodes} nodes")
        #     continue
            
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            try:
                with torch.no_grad():
                    eval_batch(model, batch, evaluator, y_true, y_pred)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"Warning: OOM during evaluation at batch {step}, skipping")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            finally:
                # Clean up batch from GPU memory
                del batch
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
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


def save_comprehensive_training_plots(train_loss_curve, train_curve, valid_curve, test_curve, 
                                    learning_rate_curve, mape_curve=None, task_type="regression", 
                                    eval_metric="rmse", output_dir="./output"):
    """
    Create comprehensive training visualization with all important curves
    """
    # Determine the number of subplots based on available data
    num_plots = 4 if mape_curve is not None else 3
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Training Analysis - {os.path.basename(output_dir)}', fontsize=16, fontweight='bold')
    
    # Plot 1: Training Loss Curve
    ax1 = axes[0, 0]
    epochs = range(1, len(train_loss_curve) + 1)
    ax1.plot(epochs, train_loss_curve, 'b-', linewidth=2, label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Curve')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Model Performance (Train/Valid/Test)
    ax2 = axes[0, 1]
    epochs = range(1, len(train_curve) + 1)
    ax2.plot(epochs, train_curve, 'g-', linewidth=2, label=f'Train {eval_metric.upper()}')
    ax2.plot(epochs, valid_curve, 'orange', linewidth=2, label=f'Valid {eval_metric.upper()}')
    ax2.plot(epochs, test_curve, 'r-', linewidth=2, label=f'Test {eval_metric.upper()}')
    
    # Mark best validation epoch
    if "classification" in task_type:
        best_epoch = np.argmax(np.array(valid_curve))
    else:
        best_epoch = np.argmin(np.array(valid_curve))
    
    ax2.axvline(x=best_epoch+1, color='purple', linestyle='--', alpha=0.7, 
                label=f'Best Valid (Epoch {best_epoch+1})')
    ax2.scatter(best_epoch+1, valid_curve[best_epoch], color='purple', s=100, zorder=5)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel(f'{eval_metric.upper()}')
    ax2.set_title(f'Model Performance ({eval_metric.upper()})')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Learning Rate Schedule
    ax3 = axes[1, 0]
    epochs = range(1, len(learning_rate_curve) + 1)
    ax3.plot(epochs, learning_rate_curve, 'm-', linewidth=2, label='Learning Rate')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.set_yscale('log')  # Use log scale for better visualization
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: MAPE Curve (for regression) or additional metrics
    ax4 = axes[1, 1]
    if mape_curve is not None and "regression" in task_type:
        epochs = range(1, len(mape_curve) + 1)
        ax4.plot(epochs, mape_curve, 'purple', linewidth=2, label='Test MAPE (%)')
        ax4.axvline(x=best_epoch+1, color='red', linestyle='--', alpha=0.7, 
                    label=f'Best Valid Epoch')
        ax4.scatter(best_epoch+1, mape_curve[best_epoch], color='red', s=100, zorder=5)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('MAPE (%)')
        ax4.set_title('Mean Absolute Percentage Error')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    else:
        # Show overfitting analysis
        if len(train_curve) > 10:  # Only if we have enough epochs
            train_smooth = np.convolve(train_curve, np.ones(5)/5, mode='valid')
            valid_smooth = np.convolve(valid_curve, np.ones(5)/5, mode='valid')
            epochs_smooth = range(3, len(train_smooth) + 3)
            
            # Calculate gap between train and validation
            gap = np.array(valid_smooth) - np.array(train_smooth)
            if "classification" in task_type:
                gap = -gap  # For classification, higher is better, so flip the gap
            
            ax4.plot(epochs_smooth, gap, 'red', linewidth=2, label='Valid-Train Gap')
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax4.fill_between(epochs_smooth, gap, 0, where=(gap > 0), 
                           color='red', alpha=0.3, label='Overfitting Region')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Performance Gap')
            ax4.set_title('Overfitting Analysis (Smoothed)')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'Insufficient data\nfor analysis', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Additional Analysis')
    
    plt.tight_layout()
    
    # Save the comprehensive plot
    plot_path = os.path.join(output_dir, 'comprehensive_training_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comprehensive training analysis saved to: {plot_path}")


def save_training_info(args, output_dir, start_time, end_time, best_epoch, best_metrics, device_info, scheduler_info):
    """
    Save comprehensive training information to README.md
    """
    readme_path = os.path.join(output_dir, "README.md")
    
    with open(readme_path, 'w') as f:
        f.write("# Training Information\n\n")
        
        # Basic Information
        f.write("## Basic Information\n")
        f.write(f"- **Training Start Time**: {start_time}\n")
        f.write(f"- **Training End Time**: {end_time}\n")
        f.write(f"- **Total Training Duration**: {end_time - start_time}\n")
        f.write(f"- **Device**: {device_info}\n")
        f.write(f"- **Output Directory**: {output_dir}\n\n")
        
        # Hyperparameters
        f.write("## Hyperparameters\n")
        f.write(f"- **Model**: {args.gnn}\n")
        f.write(f"- **Dataset**: {args.dataset}\n")
        f.write(f"- **Feature Mode**: {args.feature}\n")
        f.write(f"- **Number of Layers**: {args.num_layer}\n")
        f.write(f"- **Embedding Dimension**: {args.emb_dim}\n")
        f.write(f"- **Dropout Ratio**: {args.drop_ratio}\n")
        f.write(f"- **Batch Size**: {args.batch_size}\n")
        f.write(f"- **Learning Rate**: {args.lr}\n")
        f.write(f"- **Max Epochs**: {args.epochs}\n")
        f.write(f"- **Number of Workers**: {args.num_workers}\n\n")
        
        # Learning Rate Scheduler
        f.write("## Learning Rate Scheduler\n")
        f.write(f"- **Scheduler Type**: {scheduler_info['type']}\n")
        f.write(f"- **Mode**: {scheduler_info['mode']}\n")
        f.write(f"- **Factor**: {scheduler_info['factor']}\n")
        f.write(f"- **Patience**: {scheduler_info['patience']}\n")
        f.write(f"- **Min LR**: {scheduler_info['min_lr']}\n\n")
        
        # Best Results
        f.write("## Best Results\n")
        f.write(f"- **Best Epoch**: {best_epoch + 1}\n")
        for metric_name, metric_value in best_metrics.items():
            if isinstance(metric_value, float):
                f.write(f"- **{metric_name.capitalize()}**: {metric_value:.6f}\n")
            else:
                f.write(f"- **{metric_name.capitalize()}**: {metric_value}\n")
        f.write("\n")
        
        # File Structure
        f.write("## Output Files\n")
        f.write("```\n")
        f.write(f"{os.path.basename(output_dir)}/\n")
        f.write("├── README.md                   # This file\n")
        f.write("├── best_model.pt               # Best model checkpoint\n")
        f.write("├── training_results.json       # Detailed training results\n")
        f.write("├── training_loss.png           # Training loss curve plot\n")
        f.write("└── emergency_checkpoints/      # Emergency checkpoints (if any)\n")
        f.write("```\n\n")

        
        # Command to Reproduce
        f.write("## Command to Reproduce\n")
        f.write("```bash\n")
        
        # Create command with all current argument values
        cmd_parts = ["python src/main.py"]
        
        for arg, value in vars(args).items():
            if arg != 'output_dir':  # Don't include output_dir in reproduction command
                if isinstance(value, bool) and value:
                    cmd_parts.append(f"--{arg}")
                elif not isinstance(value, bool) and value:
                    cmd_parts.append(f"--{arg} {value}")
        
        f.write(" ".join(cmd_parts) + "\n")
        f.write("```\n")

def main():
    parser = argparse.ArgumentParser(description='GNN model')
    parser.add_argument('--gnn', type=str, default="pna", help='gnn model to use (default: pna). Available options: pna, rgcn, gat, sage, arma, film, ggnn, pan, sgn, unet, gin-virtual, gcn-virtual, gin, gcn')
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
    parser.add_argument('--output_dir', type=str, default="./output", help='directory to save all training outputs (default: ./output)')
    parser.add_argument('--max_nodes', type=int, default=5000, help='maximum number of nodes per graph to prevent OOM (default: 5000)')
    parser.add_argument('--max_edges', type=int, default=10000, help='maximum number of edges per graph to prevent OOM (default: 10000)')
    args = parser.parse_args()

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, timestamp)
    

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"All outputs will be saved to: {args.output_dir}")
    
    # Record training start time
    start_time = datetime.now()
    print(f"Training started at: {start_time}")

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    device_info = f"{device} ({torch.cuda.get_device_name(args.device) if torch.cuda.is_available() else 'CPU'})"
    print(f"Using device: {device_info}")
    
    scaler = None
    dataset = PygGraphPropPredDataset(name=args.dataset)
    if args.feature == 'simple':
        print('Using simple feature mode (only top 2 features)')
        dataset.data.x = dataset.data.x[:, :2]
        dataset.data.edge_attr = dataset.data.edge_attr[:, :2]
    else:
        print('Using full feature mode')
    
    # Simple and effective filtering for memory optimization
    print("Filtering large graphs from dataset...")
    print(f"Using max_nodes={args.max_nodes}, max_edges={args.max_edges}")
    
    # Create filtered data list
    original_size = len(dataset)
    clean_data_list = []
    for i, data in enumerate(dataset):
        if data.num_nodes <= args.max_nodes and data.num_edges <= args.max_edges:
            clean_data_list.append(data)
    
    print(f"Retained {len(clean_data_list)} / {original_size} graphs")
    
    # Create a new dataset with filtered data
    from torch_geometric.data import InMemoryDataset
    
    class FilteredDataset(InMemoryDataset):
        def __init__(self, data_list, original_dataset):
            super().__init__()
            self.data, self.slices = self.collate(data_list)
            # Copy important attributes from original dataset
            self.num_tasks = original_dataset.num_tasks
            self.task_type = original_dataset.task_type
            self.eval_metric = original_dataset.eval_metric
            # Create new split indices based on filtered data
            total_len = len(data_list)
            train_len = int(0.8 * total_len)
            valid_len = int(0.1 * total_len)
            self._split_idx = {
                'train': list(range(train_len)),
                'valid': list(range(train_len, train_len + valid_len)),
                'test': list(range(train_len + valid_len, total_len))
            }
        
        def get_idx_split(self):
            return self._split_idx
    
    # Replace dataset with filtered version
    dataset = FilteredDataset(clean_data_list, dataset)
    
    split_idx = dataset.get_idx_split()
    print(f"New split sizes - Train: {len(split_idx['train'])}, Valid: {len(split_idx['valid'])}, Test: {len(split_idx['test'])}")
    
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
    scheduler_mode = 'min' if 'regression' in dataset.task_type else 'max'
    scheduler_factor = 0.8
    scheduler_patience = 10
    scheduler_min_lr = 0.00001
    scheduler = ReduceLROnPlateau(optimizer, mode=scheduler_mode, factor=scheduler_factor, 
                                patience=scheduler_patience, min_lr=scheduler_min_lr, verbose=True)
    
    # Store scheduler info for saving
    scheduler_info = {
        'type': 'ReduceLROnPlateau',
        'mode': scheduler_mode,
        'factor': scheduler_factor,
        'patience': scheduler_patience,
        'min_lr': scheduler_min_lr
    }

    valid_curve = []
    test_curve = []
    train_curve = []
    test_predict_value = []
    test_true_value = []
    valid_predict_value = []
    valid_true_value = []

    # Add training loss curve record
    train_loss_curve = []
    # Add learning rate tracking
    learning_rate_curve = []

    for epoch in range(1, args.epochs + 1):
        print(f"===== Epoch {epoch}/{args.epochs} =====")
        
        # Clear GPU cache before each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print('Training phase...')
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current learning rate: {current_lr:.6f}')
        
        # Record current learning rate
        learning_rate_curve.append(current_lr)
    
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
            # Save best model to output directory
            model_path = os.path.join(args.output_dir, 'best_model.pt')
            print(f"Saving best model checkpoint to {model_path}")
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': test_loss,
                        'args': vars(args)  # Save training arguments
                        }, model_path)

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

        # Save results to output directory
        results_path = os.path.join(args.output_dir, 'training_results.json')
        
        result = {
            'training_info': {
                'start_time': start_time.isoformat(),
                'device': device_info,
                'hyperparameters': vars(args),
                'scheduler': scheduler_info
            },
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
                'train_loss': train_loss_curve,  # Add training loss curve
                'learning_rate': learning_rate_curve  # Add learning rate curve
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

        with open(results_path, 'w') as f:
            json.dump(result, f, indent=4)

        print(f"Results saved to: {results_path}")

        scheduler.step(valid_perf[dataset.eval_metric])


    # Record training end time and save comprehensive training info
    end_time = datetime.now()
    print(f"Training completed at: {end_time}")
    print(f"Total training duration: {end_time - start_time}")
    
    # Save comprehensive training plots (replace the simple loss plot)
    if train_loss_curve and train_curve and valid_curve and test_curve:
        mape_curve_for_plot = None
        if 'classification' not in dataset.task_type and test_predict_value:
            # Calculate MAPE curve for each epoch
            mape_curve_for_plot = []
            for epoch_preds in test_predict_value:
                if len(epoch_preds) > 0 and len(test_true_value) > 0:
                    epoch_mape = calculate_mape(test_true_value, epoch_preds)
                    mape_curve_for_plot.append(epoch_mape)
        
        save_comprehensive_training_plots(
            train_loss_curve=train_loss_curve,
            train_curve=train_curve, 
            valid_curve=valid_curve, 
            test_curve=test_curve,
            learning_rate_curve=learning_rate_curve,
            mape_curve=mape_curve_for_plot,
            task_type=dataset.task_type,
            eval_metric=dataset.eval_metric,
            output_dir=args.output_dir
        )
    
    
    # Prepare best metrics for README
    best_metrics = {
        'validation_score': valid_curve[best_val_epoch],
        'test_score': test_curve[best_val_epoch],
        'train_score': train_curve[best_val_epoch]
    }
    
    if 'classification' not in dataset.task_type and test_predict_value:
        final_test_mape = calculate_mape(test_true_value, test_predict_value[best_val_epoch])
        best_metrics['test_mape'] = f"{final_test_mape:.2f}%"
    
    # Save comprehensive training information
    save_training_info(args, args.output_dir, start_time, end_time, best_val_epoch, best_metrics, device_info, scheduler_info)
    print(f"Training information saved to: {os.path.join(args.output_dir, 'README.md')}")

if __name__ == "__main__":
    main()
