# SimpleDifferentialGNN Architecture

This document summarizes the network stack defined in `train_e2e.py` for the `SimpleDifferentialGNN` model. The model supports multiple graph backbones (`gcn`, `gin`, `rgcn`, `fast_rgcn`) but the high-level layout of layers is consistent.

## Core Encoder
- **Node Encoder:** `Linear(node_dim → hidden_dim)` projects raw node features before message passing.

## Message Passing Stack (repeated `num_layers` times)
For each layer index `i` in `0 … num_layers-1`:
- **Graph Convolution:** one of
  - `GCNConv(hidden_dim, hidden_dim)` when `gnn_type == 'gcn'`
  - `GINConv(MLP(hidden_dim → hidden_dim → hidden_dim))` when `gnn_type == 'gin'`
  - `RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=30)` when `gnn_type == 'rgcn'`
  - `FastRGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=30)` when `gnn_type == 'fast_rgcn'`
- **Normalization:** `LayerNorm(hidden_dim)`
- **Activation:** `ReLU` on all but the final message-passing block.

## Graph Readout
- **Pooling:** `global_add_pool` reduces node embeddings to a graph-level representation (summing over the batch indices).

## Prediction Heads
- **Kernel Head** *(optional; only when `differential=True` and `kernel_baseline='learned'`)*:
  - `Linear(hidden_dim → hidden_dim)` + `ReLU`
  - `Dropout(p=dropout)`
  - `Linear(hidden_dim → hidden_dim/2)` + `ReLU`
  - `Linear(hidden_dim/2 → 1)`
- **Delta Head** *(when `differential=True`)*:
  - Input dimension `4 * hidden_dim` built from `[kernel_repr, design_repr, design_repr - kernel_repr, design_repr * kernel_repr]`
  - Same MLP stack as the kernel head.
- **Design Head** *(when `differential=False`)*:
  - Same MLP stack as the kernel head but fed with the pooled design representation directly.

## Output Logic
- In differential mode, the final design prediction is `kernel_pred + delta_pred`.
- In direct mode, the design head output is returned as the prediction.

All linear layers are initialized with Xavier-uniform weights and zero biases.
