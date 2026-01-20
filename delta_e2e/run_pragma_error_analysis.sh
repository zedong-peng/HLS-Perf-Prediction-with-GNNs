#!/usr/bin/env bash

# Pragma误差分析示例（使用当前 pragma_error_analysis.py 多标签版本）

python pragma_error_analysis.py \
  --model_path /home/user/zedongpeng/workspace/HLS-Perf-Prediction-with-GNNs/delta_e2e/output/e2e_delta_dsp_20260107_014402/best_e2e_delta_dsp_model.pt \
  --design_base_dir /home/user/zedongpeng/workspace/Huggingface/forgehls_PolyBench_part_50designs \
  --kernel_base_dir /home/user/zedongpeng/workspace/Huggingface/forgehls_kernels \
  --cache_root ./graph_cache \
  --code_cache_root ./graph_cache \
  --output_dir ./pragma_error_analysis \
  --batch_size 16
