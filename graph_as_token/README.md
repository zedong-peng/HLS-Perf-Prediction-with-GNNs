# Graph-as-Token QoR (Design-Only) Training

This module mirrors the training workflow used in `GiT/Code-Verification` but targets end-to-end QoR regression on the ForgeHLS datasets. It reuses the cached kernel/design pairs produced by `delta_e2e`, encodes the design graph with a configurable GNN, and predicts the absolute QoR metrics (DSP, LUT, FF, latency) without differential targets.

## Highlights

- **Data pipeline** – `E2EDifferentialProcessor` supplies on-disk caches; `DesignQoRDataset` streams them lazily so training scales to the 100-design split.
- **Graph encoder** – `DesignQoRGNN` shares the `delta_e2e` architecture (GCN / GIN / RGCN / FastRGCN options) and pools node embeddings with sum aggregation before a multi-task regression head.
- **Metrics** – MAE / RMSE / R² are reported per metric together with global means. Results are exported for both ID (held-out test) and OOD (benchmark) sets.
- **Logging** – SwanLab integration mirrors `delta_e2e`; if SwanLab is unavailable a stub keeps the workflow dependency-free.
- **Launcher** – `run_100designs.sh` ports the ergonomics of the GiT training script (config section, ANSI logging, environment overrides) while targeting ForgeHLS 100-design + benchmark by default.

## Quick Start

```bash
cd /home/user/zedongpeng/workspace/HLS-Perf-Prediction-with-GNNs
conda activate circuitnet  # or source env/bin/activate

cd graph_as_token
EPOCHS=150 BATCH_SIZE=32 ./run_100designs.sh
```

Key environment overrides:

- `KERNEL_BASE_DIR`, `DESIGN_BASE_DIR`, `OOD_DESIGN_BASE_DIR` – dataset locations (defaults match ForgeHLS kernels + 100-design + benchmark)
- `CACHE_ROOT` – leverages `delta_e2e/graph_cache` so graphs are not rebuilt
- `OUTPUT_DIR_BASE` – base directory; a timestamped suffix is appended automatically
- `MAX_PAIRS` – limit the number of pairs for debugging (set `0` to use the full dataset)
- `USE_SWANLAB` – set to `0` to disable logging even if SwanLab is installed
- All GNN / optimization hyperparameters (`HIDDEN_DIM`, `NUM_LAYERS`, `GNN_TYPE`, `LEARNING_RATE`, etc.) can be changed via environment variables or by passing flags directly to `train_e2e.py`

## Output Artefacts

After a run you will find in `OUTPUT_DIR`:

- `best_model.pt` – checkpoint with the best validation RMSE mean
- `metrics_id_test.json`, `metrics_ood_test.json` – MAE / RMSE / R² (per metric and averaged)
- `predictions_id_test.json`, `predictions_ood_test.json` – per-design predictions with ground-truth QoR

## Roadmap

- **Graph tokens** – re-introduce a graph-token interface (GiT adapter) under a flag, allowing hybrid code/graph conditioning.
- **Differential head toggle** – optional prediction of design deltas relative to kernels while retaining the direct QoR path.
- **Feature fusion** – combine code heuristics or learned embeddings with the graph encoder for improved accuracy.

## File Overview

- `dataset.py` – dataset wrappers and cache record utilities
- `model.py` – GNN encoder + regression head
- `metrics.py` – MAE / RMSE / R² helpers
- `train_e2e.py` – training & evaluation script (SwanLab logging, ID/OOD metrics)
- `run_100designs.sh` – launcher mirroring the GiT training script experience
- `output/` – default home for experiment artefacts (created on demand)
