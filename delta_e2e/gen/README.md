# Conditional Graph Diffusion (PoC)

This directory contains a minimal, working proof-of-concept to generate design node features conditioned on kernel graph and code. It reuses existing pair cache and logging flow.

## What it does now (PoC)
- Input conditions:
  - Kernel graph node features pooled to a global vector, broadcast to design node count
  - Lightweight kernel code statistics vector (16-d) extracted from C/C++ sources and broadcast per node
- Output: generated design node feature tensor (same feature dim, variable node count via padding)
- Structure: fixed (no edge/node additions yet)
- Data: reuses `E2EDifferentialProcessor` cached kernel/design pairs
- Training: Gaussian diffusion on continuous node features with a per-node conditional denoiser (MLP)
- Logging/ckpt: SwanLab + best checkpoint saved under `runs/`

## Repo integration
- Code lives under `delta_e2e/gen/`
  - `data.py`: dataset + collate; loads cached pairs and builds conditions (graph pooled + code stats)
  - `model_diffusion.py`: conditional denoiser and Gaussian diffusion utilities
  - `train_graph_diffusion.py`: training entry
  - `sample.py`: sampling entry, saves generated node features to JSON

## Environment
- Use conda env: `circuitnet`
- For any Python command, prefer: `conda activate circuitnet && python <script>`

## Quick start
- Train (small smoke):
```bash
cd /home/user/zedongpeng/workspace/HLS-Perf-Prediction-with-GNNs
conda activate circuitnet && python -m delta_e2e.gen.train_graph_diffusion \
  --epochs 1 --batch_size 4 --max_pairs 200 --loader_workers 0
```
- Sample:
```bash
conda activate circuitnet && python -m delta_e2e.gen.sample \
  --checkpoint /home/user/zedongpeng/workspace/HLS-Perf-Prediction-with-GNNs/delta_e2e/gen/runs/attr_diffusion_*/best_attr_diffusion.pt \
  --pair_index 0
```
- Outputs:
  - Runs: `delta_e2e/gen/runs/attr_diffusion_*/`
  - Samples: `delta_e2e/gen/samples/gen_attr_*.json`

## Current limitations
- Only node feature generation; graph structure is fixed
- Kernel code condition is a heuristic 16-d statistics vector (not a learned code encoder)
- No performance-aware guidance yet (pure diffusion loss)

## Roadmap
- Stage 1: Better conditioning and controllability
  - Add target constraints: desired Δ(dsp/lut/ff/latency) or budgets as extra conditions
  - Guidance: plug in existing differential predictor (from `delta_e2e/train_e2e.py`) to guide sampling toward target metrics (classifier/classifier-free guidance)
  - Cache code features in `graph_cache` to avoid recomputation

- Stage 2: Structure-aware generation
  - Move from fixed-structure to local edit diffusion (node/edge add/remove masks)
  - Constrain connectivity and region-level semantics; support hierarchical graphs (enable `--hierarchical on` path in parsing)
  - Node alignment via anchors (names/regions) to mitigate permutation instability

- Stage 3: Stronger code conditioning
  - Replace statistics with learned encoders: GraphCodeBERT / CodeT5+ / AST/CFG GNN
  - Cross-attention between code embeddings and node embeddings in the denoiser

- Stage 4: Evaluation and E2E loop
  - Metrics: validity rate, edit distance, attr MAE/RMSE, diversity
  - Performance verification: run differential predictor on generated graphs; sample subset for real HLS synth
  - Close the loop: generate pragma suggestions and (optionally) patch code, then re-evaluate

- Stage 5: Engineering & scaling
  - Mixed precision (AMP), gradient checkpointing, distributed training
  - Config files (YAML) and CLI parity with existing training scripts
  - CI checks and unit tests for data, model, and sampling

## Notes
- Default data roots match the project rules:
  - Kernel: `/home/user/zedongpeng/workspace/HLS-Perf-Prediction-with-GNNs/Graphs/forgehls_kernels/kernels/`
  - Design: `/home/user/zedongpeng/workspace/Huggingface/forgehls_lite_10designs/`
- You can override via CLI flags in the training/sampling scripts. 