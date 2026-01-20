#!/usr/bin/env bash

# 8:1:1 误差分析，复用训练配置，输出 kernel/设计 MAPE/MAE 统计

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

python train_split_error_analysis.py \
  --model_path /home/user/zedongpeng/workspace/HLS-Perf-Prediction-with-GNNs/delta_e2e/output/e2e_delta_dsp_20260107_014402/best_e2e_delta_dsp_model.pt \
  --design_base_dir /home/user/zedongpeng/workspace/Huggingface/forgehls_PolyBench_part_500designs \
  --kernel_base_dir /home/user/zedongpeng/workspace/Huggingface/forgehls_kernels \
  --cache_root ./graph_cache \
  --batch_size 16 \
  --seed 42
