#!/usr/bin/env python3
"""
Modularized GNN Training Script with SwanLab Integration
========================================================

This script provides a complete training pipeline for Graph Neural Networks (GNNs)
with experiment tracking via SwanLab and comprehensive result visualization.

Author: Zedong Peng
Date: 2025-09-05
"""

import torch
from torch_geometric.data import DataLoader, InMemoryDataset
import torch.optim as optim
from torch_geometric.utils import degree
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm
import argparse
import numpy as np
import json
import operator
from functools import reduce
import os
from datetime import datetime
import matplotlib.pyplot as plt
import gc
from typing import Dict, Any, Optional, List, Tuple

from dataset_pyg import PygGraphPropPredDataset
from evaluate import Evaluator


# ============================================================================
# Loss Functions and Criteria
# ============================================================================

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()


# ============================================================================
# SwanLab Logger Module
# ============================================================================

class SwanLabLogger:
    """SwanLab logger for experiment tracking"""
    
    def __init__(self, use_swanlab: bool = False, project: str = "gnn-hls-prediction", 
                 experiment_name: str = None, config: Dict[str, Any] = None):
        self.use_swanlab = use_swanlab
        self.swanlab = None
        
        if self.use_swanlab:
            try:
                import swanlab
                self.swanlab = swanlab
                
                # Initialize SwanLab run
                self.run = swanlab.init(
                    project=project,
                    experiment_name=experiment_name,
                    config=config
                )
                print(f"SwanLab initialized - Project: {project}, Experiment: {experiment_name}")
            except ImportError:
                print("Warning: SwanLab not installed. Install with: pip install swanlab")
                self.use_swanlab = False
            except Exception as e:
                print(f"Warning: Failed to initialize SwanLab: {e}")
                self.use_swanlab = False
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to SwanLab"""
        if self.use_swanlab and self.swanlab:
            try:
                self.swanlab.log(metrics, step=step)
            except Exception as e:
                print(f"Warning: Failed to log metrics to SwanLab: {e}")
    
    def log_model_summary(self, model, input_shape: tuple = None):
        """Log model summary to SwanLab"""
        if self.use_swanlab and self.swanlab:
            try:
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                model_info = {
                    "model/total_parameters": total_params,
                    "model/trainable_parameters": trainable_params,
                    "model/size_mb": total_params * 4 / 1024 / 1024,
                }
                
                self.swanlab.log(model_info)
                print(f"Model summary logged - Total params: {total_params:,}, Trainable: {trainable_params:,}")
            except Exception as e:
                print(f"Warning: Failed to log model summary to SwanLab: {e}")
    
    def log_dataset_info(self, dataset_info: Dict[str, Any]):
        """Log dataset information to SwanLab"""
        if self.use_swanlab and self.swanlab:
            try:
                formatted_info = {f"dataset/{k}": v for k, v in dataset_info.items()}
                self.swanlab.log(formatted_info)
            except Exception as e:
                print(f"Warning: Failed to log dataset info to SwanLab: {e}")
    
    def log_training_curves(self, epoch: int, train_loss: float, train_metrics: Dict[str, float],
                           valid_metrics: Dict[str, float], test_metrics: Dict[str, float],
                           learning_rate: float):
        """Log training curves and metrics for each epoch"""
        if self.use_swanlab and self.swanlab:
            try:
                log_dict = {
                    "train/loss": train_loss,
                    "train/learning_rate": learning_rate,
                }
                
                # Add train metrics
                for metric, value in train_metrics.items():
                    log_dict[f"train/{metric}"] = value
                
                # Add validation metrics
                for metric, value in valid_metrics.items():
                    log_dict[f"valid/{metric}"] = value
                
                # Add test metrics
                for metric, value in test_metrics.items():
                    log_dict[f"test/{metric}"] = value
                
                self.swanlab.log(log_dict, step=epoch)
            except Exception as e:
                print(f"Warning: Failed to log training curves to SwanLab: {e}")
    
    def log_best_results(self, best_epoch: int, best_metrics: Dict[str, Any]):
        """Log best results summary"""
        if self.use_swanlab and self.swanlab:
            try:
                summary_dict = {"best/epoch": best_epoch}
                for metric, value in best_metrics.items():
                    summary_dict[f"best/{metric}"] = value
                
                self.swanlab.log(summary_dict)
                print(f"Best results logged - Epoch: {best_epoch}")
            except Exception as e:
                print(f"Warning: Failed to log best results to SwanLab: {e}")
    
    def log_image(self, image_path: str, name: str = None):
        """Log image to SwanLab"""
        if self.use_swanlab and self.swanlab:
            try:
                if os.path.exists(image_path):
                    image_name = name or os.path.basename(image_path).replace('.png', '')
                    self.swanlab.log({f"plots/{image_name}": self.swanlab.Image(image_path)})
                    print(f"Image logged: {image_name}")
            except Exception as e:
                print(f"Warning: Failed to log image to SwanLab: {e}")
    
    def finish(self):
        """Finish SwanLab run"""
        if self.use_swanlab and self.swanlab:
            try:
                self.swanlab.finish()
                print("SwanLab run finished successfully")
            except Exception as e:
                print(f"Warning: Failed to finish SwanLab run: {e}")


# ============================================================================
# Configuration Management Module
# ============================================================================

def get_parser():
    """Get argument parser with all training configurations"""
    parser = argparse.ArgumentParser(description='GNN model training')
    
    # Model parameters
    parser.add_argument('--gnn', type=str, default="pna", 
                       help='gnn model to use (default: pna). Available options: pna, rgcn, gat, sage, arma, film, ggnn, pan, sgn, unet, gin-virtual, gcn-virtual, gin, gcn')
    parser.add_argument('--num_layer', type=int, default=5, 
                       help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300, 
                       help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--drop_ratio', type=float, default=0.5, 
                       help='dropout ratio (default: 0.5)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='batch size for training and validation')
    parser.add_argument('--lr', type=float, default=0.001, 
                       help='learning rate')
    parser.add_argument('--epochs', type=int, default=300, 
                       help='maximum number of epochs to train (default: 300)')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default="dfg_dsp_binary", 
                       help='dataset name (default: dfg_dsp_binary)')
    parser.add_argument('--feature', type=str, default="full", 
                       help='full feature or simple feature')
    parser.add_argument('--max_nodes', type=int, default=5000, 
                       help='maximum number of nodes per graph to prevent OOM (default: 5000)')
    parser.add_argument('--max_edges', type=int, default=10000, 
                       help='maximum number of edges per graph to prevent OOM (default: 10000)')
    
    # System parameters
    parser.add_argument('--device', type=int, default=0, 
                       help='which gpu to use if any (default: 0)')
    parser.add_argument('--num_workers', type=int, default=0, 
                       help='number of workers (default: 0)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default="./output", 
                       help='directory to save all training outputs (default: ./output)')
    
    # SwanLab parameters
    parser.add_argument('--use_swanlab', action='store_true', 
                       help='use SwanLab for experiment tracking')
    parser.add_argument('--swanlab_project', type=str, default="gnn-hls-prediction", 
                       help='SwanLab project name')
    parser.add_argument('--experiment_name', type=str, default=None, 
                       help='experiment name for SwanLab')
    
    return parser


def setup_config(args):
    """Setup configuration with timestamp and output directory"""
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = f"{args.gnn}_{args.dataset}_{timestamp}"
    
    return args


def get_scheduler_config():
    """Get learning rate scheduler configuration"""
    return {
        'factor': 0.8,
        'patience': 10,
        'min_lr': 0.00001
    }


def get_model_class(gnn_type):
    """Import and return the appropriate model class"""
    if gnn_type == "pna":
        from pna import Net
    elif gnn_type == "rgcn":
        from rgcn import Net
    elif gnn_type == "gat":
        from gat import Net
    elif gnn_type == "sage":
        from sage import Net
    elif gnn_type == "arma":
        from ARMA import Net
    elif gnn_type == "film":
        from film import Net
    elif gnn_type == "ggnn":
        from ggnn import Net
    elif gnn_type == "pan":
        from pan import Net
    elif gnn_type == "sgn":
        from sgn import Net
    elif gnn_type == "unet":
        from unet import Net
    elif gnn_type in ['gin', 'gin-virtual', 'gcn', 'gcn-virtual']:
        from gnn import GNN as Net
    else:
        raise ValueError(f'--gnn value not supported: {gnn_type}. Available options: pna, rgcn, gat, sage, arma, film, ggnn, pan, sgn, unet, gin-virtual, gcn-virtual, gin, gcn')
    
    return Net


def get_model_args(args, dataset, split_idx=None):
    """Get model arguments based on configuration"""
    model_args = {
        'num_tasks': dataset.num_tasks, 
        'num_layer': args.num_layer, 
        'emb_dim': args.emb_dim, 
        'drop_ratio': args.drop_ratio
    }
    
    # Add specific arguments for different models
    if args.gnn == "pna" and split_idx is not None:
        max_degree = max(degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long).max().item() 
                        for data in dataset[split_idx["train"]])
        deg = torch.zeros(max_degree + 1, dtype=torch.long)
        for data in dataset[split_idx["train"]]:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())
        model_args['deg'] = deg
    elif args.gnn == "gat":
        model_args['heads'] = 8
    elif args.gnn in ['gin', 'gin-virtual', 'gcn', 'gcn-virtual']:
        model_args['gnn_type'] = args.gnn.split('-')[0]
        model_args['virtual_node'] = '-virtual' in args.gnn
    
    return model_args


# ============================================================================
# Utility Functions Module
# ============================================================================

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error (MAPE)"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def setup_device(device_id: int) -> Tuple[torch.device, str]:
    """Setup and return device information"""
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    device_info = f"{device} ({torch.cuda.get_device_name(device_id) if torch.cuda.is_available() else 'CPU'})"
    return device, device_info


def setup_workers(num_workers: int) -> int:
    """Setup number of workers for data loading"""
    if num_workers > 0:
        print(f"Using specified num_workers={num_workers}")
        return num_workers
    else:
        workers = max(2, os.cpu_count() // 2)
        print(f"Automatically set num_workers={workers} (half of CPU cores)")
        return workers


class FilteredDataset(InMemoryDataset):
    """Filtered dataset with random split"""
    
    def __init__(self, data_list, original_dataset):
        super().__init__()
        self.data, self.slices = self.collate(data_list)
        # Copy important attributes from original dataset
        self.num_tasks = original_dataset.num_tasks
        self.eval_metric = original_dataset.eval_metric
        
        # Random split instead of sequential
        total_len = len(data_list)
        indices = np.random.permutation(total_len)
        
        train_len = int(0.8 * total_len)
        valid_len = int(0.1 * total_len)
        
        self._split_idx = {
            'train': indices[:train_len].tolist(),
            'valid': indices[train_len:train_len + valid_len].tolist(),
            'test': indices[train_len + valid_len:].tolist()
        }
    
    def get_idx_split(self):
        return self._split_idx


def filter_dataset(dataset, max_nodes: int, max_edges: int):
    """Filter dataset to remove large graphs"""
    print("Filtering large graphs from dataset...")
    print(f"Using max_nodes={max_nodes}, max_edges={max_edges}")
    
    original_size = len(dataset)
    clean_data_list = []
    for i, data in enumerate(dataset):
        if data.num_nodes <= max_nodes and data.num_edges <= max_edges:
            clean_data_list.append(data)
    
    print(f"Retained {len(clean_data_list)} / {original_size} graphs")
    return FilteredDataset(clean_data_list, dataset)


# ============================================================================
# Training Module
# ============================================================================

def train_batch(model, batch, optimizer, scaler=None):
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
    
    # Unified regression loss
    loss = reg_criterion(pred, target)
    
    # Store loss value before clearing variables
    loss_value = loss.item()
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Clean up intermediate variables
    del pred, target, is_labeled, loss
    
    return loss_value


def eval_batch(model, batch, evaluator, y_true, y_pred):
    """Process evaluation logic for a single batch"""
    pred = model(batch)
    if pred.shape[0] != batch.y.shape[0]:
        print(f"Warning: Skipping mismatched shapes during evaluation. Pred shape: {pred.shape}, Label shape: {batch.y.shape}")
        return
    y_true.append(batch.y.view(pred.shape).detach().cpu())
    y_pred.append(pred.detach().cpu())
    
    # Clean up predictions
    del pred


def train_epoch(model, device, loader, optimizer, scaler=None):
    """Train model for one epoch"""
    model.train()
    total_loss = 0
    batch_count = 0
    loss_log_interval = 100
    
    for step, batch in enumerate(tqdm(loader, desc="Training")):
        if not hasattr(batch, 'x') or not hasattr(batch, 'batch') or batch.x.shape[0] == 0 or batch.batch.shape[0] == 0:
            print(f"Warning: Skipping empty batch. Batch {step}")
            continue
            
        batch = batch.to(device)
        if batch.x.shape[0] == 1 or (batch.batch.shape[0] > 0 and batch.batch[-1] == 0):
            pass
        else:
            loss = train_batch(model, batch, optimizer, scaler)
            if loss is not None:
                total_loss += loss
                batch_count += 1
                
                # Print average loss periodically
                if (step + 1) % loss_log_interval == 0:
                    avg_loss = total_loss / batch_count
                    print(f"Batch {step+1}/{len(loader)}, Average Loss: {avg_loss:.6f}")
                    
                    # Clean GPU cache periodically
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    
    # Output overall training loss
    if batch_count > 0:
        epoch_loss = total_loss / batch_count
        print(f"Training completed. Epoch average loss: {epoch_loss:.6f}")
        return epoch_loss
    return float('inf')


def evaluate_model(model, device, loader, evaluator):
    """Evaluate model on given data loader"""
    model.eval()
    y_true = []
    y_pred = []
    
    for step, batch in enumerate(tqdm(loader, desc="Evaluation")):
        if not hasattr(batch, 'x') or not hasattr(batch, 'batch') or batch.x.shape[0] == 0 or batch.batch.shape[0] == 0:
            print(f"Warning: Skipping empty batch during evaluation. Batch {step}")
            continue
        
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                eval_batch(model, batch, evaluator, y_true, y_pred)
            if step % 100 == 0:
                print(f"Processed {step+1}/{len(loader)} batches in evaluation")

    if len(y_true) == 0 or len(y_pred) == 0:
        print("Warning: No valid prediction results. Please check dataset and model.")
        return {"rmse": float("inf")}, np.array([]), np.array([])
    
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return evaluator.eval(input_dict), y_true, y_pred


# ============================================================================
# Visualization Module
# ============================================================================

def save_comprehensive_training_plots(train_loss_curve, train_curve, valid_curve, test_curve,
                                    learning_rate_curve, mape_curve=None,
                                    eval_metric="rmse", output_dir="./output"):
    """Create comprehensive training visualization with all important curves"""
    # Determine the number of subplots based on available data
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
    # For regression (lower is better)
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
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: MAPE Curve or Overfitting Analysis
    ax4 = axes[1, 1]
    if mape_curve is not None:
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
        if len(train_curve) > 10:
            train_smooth = np.convolve(train_curve, np.ones(5)/5, mode='valid')
            valid_smooth = np.convolve(valid_curve, np.ones(5)/5, mode='valid')
            epochs_smooth = range(3, len(train_smooth) + 3)
            
            # Calculate gap between train and validation (regression)
            gap = np.array(valid_smooth) - np.array(train_smooth)
            
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
    return plot_path


def save_training_info(args, output_dir, start_time, end_time, best_epoch, best_metrics, device_info, scheduler_info):
    """Save comprehensive training information to README.md"""
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
        f.write("├── comprehensive_training_analysis.png # Training visualization\n")
        f.write("└── emergency_checkpoints/      # Emergency checkpoints (if any)\n")
        f.write("```\n\n")
        
        # Command to Reproduce
        f.write("## Command to Reproduce\n")
        f.write("```bash\n")
        
        # Create command with all current argument values
        cmd_parts = ["python src/train.py"]
        
        for arg, value in vars(args).items():
            if arg != 'output_dir':  # Don't include output_dir in reproduction command
                if isinstance(value, bool) and value:
                    cmd_parts.append(f"--{arg}")
                elif not isinstance(value, bool) and value:
                    cmd_parts.append(f"--{arg} {value}")
        
        f.write(" ".join(cmd_parts) + "\n")
        f.write("```\n")


# ============================================================================
# Main Trainer Class
# ============================================================================

class GNNTrainer:
    """Main trainer class that orchestrates the entire training process"""
    
    def __init__(self, args):
        self.args = args
        self.device, self.device_info = setup_device(args.device)
        self.start_time = datetime.now()
        
        # Initialize logger
        logger_config = vars(args) if args.use_swanlab else None
        self.logger = SwanLabLogger(
            use_swanlab=args.use_swanlab,
            project=args.swanlab_project,
            experiment_name=args.experiment_name,
            config=logger_config
        )
        
        # Initialize tracking variables
        self.train_loss_curve = []
        self.learning_rate_curve = []
        self.mape_curve = []
        self.valid_curve = []
        self.test_curve = []
        self.train_curve = []
        
        print(f"Trainer initialized - Using device: {self.device_info}")
        print(f"All outputs will be saved to: {args.output_dir}")
    
    def setup_dataset(self):
        """Setup and prepare dataset"""
        dataset = PygGraphPropPredDataset(name=self.args.dataset)
        
        if self.args.feature == 'simple':
            print('Using simple feature mode (only top 2 features)')
            dataset.data.x = dataset.data.x[:, :2]
            dataset.data.edge_attr = dataset.data.edge_attr[:, :2]
        else:
            print('Using full feature mode')
        
        # Filter dataset
        self.dataset = filter_dataset(dataset, self.args.max_nodes, self.args.max_edges)
        self.split_idx = self.dataset.get_idx_split()
        
        print(f"Split sizes - Train: {len(self.split_idx['train'])}, Valid: {len(self.split_idx['valid'])}, Test: {len(self.split_idx['test'])}")
        
        # Log dataset info
        dataset_info = {
            "name": self.args.dataset,
            "feature_mode": self.args.feature,
            "train_size": len(self.split_idx['train']),
            "valid_size": len(self.split_idx['valid']),
            "test_size": len(self.split_idx['test']),
            "max_nodes": self.args.max_nodes,
            "max_edges": self.args.max_edges
        }
        self.logger.log_dataset_info(dataset_info)
    
    def setup_model(self):
        """Setup model, optimizer, and scheduler"""
        # Get model arguments
        model_args = get_model_args(self.args, self.dataset, self.split_idx)
        
        # Create model
        Net = get_model_class(self.args.gnn)
        self.model = Net(**model_args).to(self.device)
        
        # Log model summary
        self.logger.log_model_summary(self.model)
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        
        # Setup scheduler
        scheduler_config = get_scheduler_config()
        # Regression: lower is better
        scheduler_mode = 'min'
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode=scheduler_mode, 
            factor=scheduler_config['factor'],
            patience=scheduler_config['patience'], 
            min_lr=scheduler_config['min_lr']
        )
        
        # Store scheduler info
        self.scheduler_info = {
            'type': 'ReduceLROnPlateau',
            'mode': scheduler_mode,
            'factor': scheduler_config['factor'],
            'patience': scheduler_config['patience'],
            'min_lr': scheduler_config['min_lr']
        }
    
    def setup_data_loaders(self):
        """Setup data loaders"""
        pin_memory = torch.cuda.is_available()
        num_workers = setup_workers(self.args.num_workers)
        
        self.train_loader = DataLoader(
            self.dataset[self.split_idx["train"]], 
            batch_size=self.args.batch_size, 
            shuffle=True, 
            num_workers=num_workers, 
            pin_memory=pin_memory
        )
        self.valid_loader = DataLoader(
            self.dataset[self.split_idx["valid"]], 
            batch_size=self.args.batch_size, 
            shuffle=False, 
            num_workers=num_workers, 
            pin_memory=pin_memory
        )
        self.test_loader = DataLoader(
            self.dataset[self.split_idx["test"]], 
            batch_size=self.args.batch_size, 
            shuffle=False, 
            num_workers=num_workers, 
            pin_memory=pin_memory
        )
        
        self.evaluator = Evaluator(self.args.dataset)
    
    def train_one_epoch(self, epoch):
        """Train for one epoch and evaluate"""
        print(f"===== Epoch {epoch}/{self.args.epochs} =====")
        
        # Clean GPU cache at epoch start
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Get current learning rate
        try:
            current_lr = self.scheduler.get_last_lr()[0]
        except Exception:
            current_lr = self.optimizer.param_groups[0]['lr']
        print(f'Current learning rate: {current_lr:.6f}')
        self.learning_rate_curve.append(current_lr)
        
        # Training phase
        print('Training phase...')
        epoch_loss = train_epoch(self.model, self.device, self.train_loader, self.optimizer)
        self.train_loss_curve.append(epoch_loss)
        
        # Evaluation phase
        print('Evaluation phase...')
        with torch.no_grad():
            train_perf, train_true, train_pred = evaluate_model(self.model, self.device, self.train_loader, self.evaluator)
            valid_perf, v_true, v_pred = evaluate_model(self.model, self.device, self.valid_loader, self.evaluator)
            test_perf, t_true, t_pred = evaluate_model(self.model, self.device, self.test_loader, self.evaluator)
        
        # Calculate and log MAPE (regression)
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
        
        # Record MAPE for curve plotting
        self.mape_curve.append(test_mape)

        print({'Train': {'rmse': train_perf[self.dataset.eval_metric], 'mape': f'{train_mape:.2f}%'},
               'Validation': {'rmse': valid_perf[self.dataset.eval_metric], 'mape': f'{valid_mape:.2f}%'},
               'Test': {'rmse': test_perf[self.dataset.eval_metric], 'mape': f'{test_mape:.2f}%'}})
        
        # Record curves
        self.train_curve.append(train_perf[self.dataset.eval_metric])
        self.valid_curve.append(valid_perf[self.dataset.eval_metric])
        self.test_curve.append(test_perf[self.dataset.eval_metric])
        
        # Log to SwanLab
        self.logger.log_training_curves(
            epoch=epoch,
            train_loss=epoch_loss,
            train_metrics=train_perf,
            valid_metrics=valid_perf,
            test_metrics=test_perf,
            learning_rate=current_lr
        )
        
        # Check if this is the best epoch and save model
        is_best = self._is_best_epoch()
        if is_best:
            self._save_best_model(epoch, valid_perf, train_true, train_pred, v_true, v_pred, t_true, t_pred)
        
        # Clean up memory
        del train_true, train_pred, v_true, v_pred, t_true, t_pred
        gc.collect()
        
        # Update scheduler
        self.scheduler.step(valid_perf[self.dataset.eval_metric])
        
        # Print epoch summary
        best_val_epoch = self._get_best_epoch_idx()
        print(f'Best validation score (epoch {best_val_epoch+1}): {self.valid_curve[best_val_epoch]:.6f}')
        print(f'Test score at best validation epoch: {self.test_curve[best_val_epoch]:.6f}')
        
        return epoch_loss, train_perf, valid_perf, test_perf
    
    def _is_best_epoch(self):
        """Check if current epoch is the best based on validation performance"""
        # Regression: lower is better
        return self.valid_curve[-1] <= min(self.valid_curve)
    
    def _get_best_epoch_idx(self):
        """Get the index of the best epoch"""
        # Regression: lower is better
        return np.argmin(np.array(self.valid_curve))
    
    def _save_best_model(self, epoch, valid_perf, train_true, train_pred, v_true, v_pred, t_true, t_pred):
        """Save the best model checkpoint and predictions"""
        # Save best epoch predictions
        self.best_test_pred = reduce(operator.add, t_pred.tolist()).copy()
        self.best_valid_pred = reduce(operator.add, v_pred.tolist()).copy()
        
        # Save model
        model_path = os.path.join(self.args.output_dir, 'best_model.pt')
        print(f"Saving best model checkpoint to {model_path}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': valid_perf[self.dataset.eval_metric],
            'args': vars(self.args)
        }, model_path)
    
    def save_final_results(self):
        """Save final training results and generate visualizations"""
        end_time = datetime.now()
        print(f"Training completed at: {end_time}")
        print(f"Total training duration: {end_time - self.start_time}")
        
        # Get best epoch
        best_val_epoch = self._get_best_epoch_idx()
        
        # Prepare final results
        result = {
            'training_info': {
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration': str(end_time - self.start_time),
                'device': self.device_info,
                'hyperparameters': vars(self.args),
                'scheduler': self.scheduler_info
            },
            'metrics': {
                'val': self.valid_curve[best_val_epoch],
                'test': self.test_curve[best_val_epoch],
                'train': self.train_curve[best_val_epoch]
            },
            'best_epoch': int(best_val_epoch),
            'curves': {
                'train': self.train_curve,
                'valid': self.valid_curve,
                'test': self.test_curve,
                'train_loss': self.train_loss_curve,
                'learning_rate': self.learning_rate_curve
            }
        }
        
        # Add MAPE results and error stats (regression)
        if hasattr(self, 'best_test_pred') and hasattr(self, 'final_test_true'):
            if self.final_test_true and self.best_test_pred:
                final_test_mape = calculate_mape(self.final_test_true, self.best_test_pred)
                result['metrics']['test_mape'] = final_test_mape
                print(f'Final test MAPE: {final_test_mape:.2f}%')
                
                # Calculate error statistics
                errors = [abs(pred - true) for pred, true in zip(self.best_test_pred, self.final_test_true)]
                result['error_stats'] = {
                    'min': min(errors),
                    'max': max(errors),
                    'mean': sum(errors) / len(errors),
                    'median': sorted(errors)[len(errors) // 2],
                    'percentile_90': sorted(errors)[int(len(errors) * 0.9)]
                }
        # Add MAPE curve to results
        result['curves']['mape'] = self.mape_curve
        
        # Save results JSON
        results_path = os.path.join(self.args.output_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(result, f, indent=4)
        print(f"Results saved to: {results_path}")
        
        # Generate and save plots
        plot_path = save_comprehensive_training_plots(
            train_loss_curve=self.train_loss_curve,
            train_curve=self.train_curve, 
            valid_curve=self.valid_curve, 
            test_curve=self.test_curve,
            learning_rate_curve=self.learning_rate_curve,
            mape_curve=self.mape_curve if self.mape_curve else None,
            eval_metric=self.dataset.eval_metric,
            output_dir=self.args.output_dir
        )
        
        # Log plot to SwanLab
        self.logger.log_image(plot_path, "comprehensive_training_analysis")
        
        # Prepare best metrics for README
        best_metrics = {
            'validation_score': self.valid_curve[best_val_epoch],
            'test_score': self.test_curve[best_val_epoch],
            'train_score': self.train_curve[best_val_epoch]
        }
        
        if hasattr(self, 'best_test_pred') and hasattr(self, 'final_test_true'):
            if self.final_test_true and self.best_test_pred:
                final_test_mape = calculate_mape(self.final_test_true, self.best_test_pred)
                best_metrics['test_mape'] = f"{final_test_mape:.2f}%"
        
        # Log best results to SwanLab
        self.logger.log_best_results(best_val_epoch, best_metrics)
        
        # Save training info
        save_training_info(self.args, self.args.output_dir, self.start_time, end_time, 
                          best_val_epoch, best_metrics, self.device_info, self.scheduler_info)
        print(f"Training information saved to: {os.path.join(self.args.output_dir, 'README.md')}")
    
    def train(self):
        """Main training loop"""
        print(f"Training started at: {self.start_time}")
        
        # Setup components
        self.setup_dataset()
        self.setup_model()
        self.setup_data_loaders()
        
        # Initialize final test true values storage
        self.final_test_true = None
        
        # Training loop
        for epoch in range(1, self.args.epochs + 1):
            epoch_loss, train_perf, valid_perf, test_perf = self.train_one_epoch(epoch)
            
            # Store true values only once (they don't change across epochs)
            if self.final_test_true is None:
                with torch.no_grad():
                    _, t_true, _ = evaluate_model(self.model, self.device, self.test_loader, self.evaluator)
                    self.final_test_true = reduce(operator.add, t_true.tolist())
        
        # Save final results and finish
        self.save_final_results()
        self.logger.finish()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main function to run the training pipeline"""
    # Parse arguments and setup configuration
    parser = get_parser()
    args = parser.parse_args()
    args = setup_config(args)
    
    # Create and run trainer
    trainer = GNNTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
