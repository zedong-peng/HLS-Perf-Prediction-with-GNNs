# 废案 NOT WORKING

# Conditional Graph Diffusion

End-to-end diffusion pipeline that generates complete design graphs (nodes + adjacency + edge attributes) from kernel graphs and design code statistics, then evaluates the generated graph quality and feeds it into the existing QoR predictor.

## Components
- `data.py` – loads cached kernel/design pairs, builds dense edge tensors, extracts code features from each design directory, and exposes masks for variable-size graphs.
- `model_diffusion.py` – joint node/edge denoiser and Gaussian diffusion utilities.
- `train_graph_diffusion.py` – training entry point with joint node/edge losses and optional SwanLab logging.
- `inference.py` – helpers for loading checkpoints and sampling conditioned graphs.
- `metrics.py` – structural and feature-level metrics (MAE/RMSE, precision/recall/F1, degree/density gaps, edge-attribute scores).
- `sample.py` – generates graphs, reports quality metrics, and exports JSON artefacts.
- `predict_qor_with_generated.py` – plugs a generated graph into the delta_e2e QoR model to estimate resource/latency metrics.

## Quick Start
Activate the project environment before running any command:
```bash
cd /home/user/zedongpeng/workspace/HLS-Perf-Prediction-with-GNNs
conda activate circuitnet
```

### Train the diffusion model
```bash
python -m delta_e2e.gen.train_graph_diffusion \
  --epochs 5 \
  --batch_size 4 \
  --max_pairs 200 \
  --loader_workers 0
```
Checkpoints are stored under `delta_e2e/gen/runs/graph_diffusion_*`. Pass `--use_swanlab` to enable online logging (requires SwanLab and network access).

### Train node-only diffusion
When you only want to denoise node features (no adjacency/edge channels), enable the dedicated flag:
```bash
python -m delta_e2e.gen.train_graph_diffusion \
  --nodes_only \
  --epochs 5 \
  --batch_size 4
```
This automatically masks out edge losses/metrics while still emitting node-quality scores during evaluation.

### Sample a design graph and inspect metrics
```bash
python -m delta_e2e.gen.sample \
  --checkpoint delta_e2e/gen/runs/graph_diffusion_*/best_graph_diffusion.pt \
  --pair_index 0 \
  --edge_threshold 0.55 \
  --include_target
```
Outputs live in `delta_e2e/gen/samples/gen_graph_*.json` and include:
- generated node features, adjacency logits/probabilities, and edge attributes
- sparse edge list (`edge_index`, `edge_attr`)
- graph quality metrics computed against the reference design graph

### Predict QoR with a generated graph
```bash
python -m delta_e2e.gen.predict_qor_with_generated \
  --diffusion_ckpt delta_e2e/gen/runs/graph_diffusion_*/best_graph_diffusion.pt \
  --qor_ckpt delta_e2e/output/best_e2e_delta_dsp_model.pt \
  --pair_index 0 \
  --edge_threshold 0.55
```
The script samples a graph, converts it into a `torch_geometric.data.Data` object, and runs the saved delta_e2e GNN checkpoint to estimate QoR. A JSON summary containing predicted/ground-truth metrics and graph quality scores is written to `delta_e2e/gen/runs/qor_eval_*.json`.

## Graph Quality Metrics
`metrics.py` computes:
- **node_feature_mae / rmse / cosine** – compare generated and reference node embeddings under the valid-node mask
- **adjacency_mae / precision / recall / f1** – evaluate binary structure after thresholding `sigmoid(logits)` at the requested edge threshold
- **degree_l1, density_abs** – capture distribution-level deviations
- **edge_attr_mae / cosine** – only when edge attributes exist; evaluated on edges present in the ground-truth graph

Metrics are returned as plain floats (or `null` when not applicable) and logged by both `sample.py` and `predict_qor_with_generated.py`.

## Implementation Notes
- Code features are extracted from the design directory (recursively walking for C/C++ files) to align with downstream QoR usage.
- Dense edge tensors contain a binary adjacency channel plus original edge attributes; padding masks keep diffusion/training numerically stable for variable graph sizes.
- Generation and QoR inference share the `inference.py` helpers to ensure consistent post-processing (masking, thresholding, sparse conversion).

## Limitations & Next Steps
- Edge sampling currently thresholds post-hoc; exploring differentiable relaxation or classifier-free guidance can improve graph validity.
- Only design-level code statistics are used; replacing them with learned code encoders (GraphCodeBERT/CodeT5) remains future work.
- QoR conditioning is one-shot; iterative correction or joint training with the QoR model is not yet implemented.

Override data roots with `--kernel_base_dir` / `--design_base_dir` if needed; defaults match the repository layout.
