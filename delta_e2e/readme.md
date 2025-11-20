# SimpleDifferentialGNN Architecture

This document summarizes the network stack defined in `train_e2e.py` for the `SimpleDifferentialGNN` model. The model supports multiple graph backbones (`gcn`, `gin`, `rgcn`, `fast_rgcn`) but the high-level layout of layers is consistent.

好的，这里是一个更**简洁、面向组会汇报的版本**（去掉 `std`、`p90`，仅保留最有信息量的指标：mean / median / p95 / max）：

---

# 🔧 ForgeHLS Resource Statistics (Simplified)

| Metric      | Type   |     Mean | Median |     p95 |       Max |
| :---------- | :----- | -------: | -----: | ------: | --------: |
| **DSP**     | Kernel |     17.7 |      0 |      35 |      3.6K |
|             | Design |     91.8 |      0 |      80 |      225K |
|             | Δ      |     74.0 |      0 |      35 |      225K |
| **LUT**     | Kernel |     2.7K |    194 |    8.2K |      394K |
|             | Design |    13.4K |    577 |   29.3K |     14.8M |
|             | Δ      |    10.6K |    236 |   18.5K |     14.8M |
| **FF**      | Kernel |    1.39K |     45 |   5.97K |       55K |
|             | Design |    6.81K |    110 |   17.0K |      6.5M |
|             | Δ      |    5.42K |     25 |   9.45K |      6.5M |
| **Latency** | Kernel | 4.15×10⁸ |   1.0K | 5.2×10⁶ |  6.0×10¹⁰ |
|             | Design | 6.80×10⁸ |   1.0K | 2.4×10⁷ |  1.1×10¹¹ |
|             | Δ      | 2.65×10⁸ |     −1 | 6.0×10⁶ | 1.05×10¹¹ |

---

### 💡 Quick Takeaways

* **Resources (DSP/LUT/FF)** increase **3–10×** after HLS optimization (Design vs Kernel).
* **Δ distributions** are long-tailed but mostly small; many cases remain near-zero (sparse changes).
* **Latency** spans over **10 orders of magnitude**, dominated by a few extremely long-running designs — normalization or log-scale modeling is essential.
* **Median values near 0** show most kernels are lightweight; heavy outliers dominate resource variance.


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
