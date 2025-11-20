#!/usr/bin/env bash
# Frozen Qwen coder QoR regression on the ForgeHLS 10-design subset.

set -euo pipefail

PROJECT_ROOT="$(dirname "$(realpath "$0")")"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/forgehls_lite_10designs"
CACHE_DIR="${PROJECT_ROOT}/cache"
mkdir -p "${OUTPUT_DIR}" "${CACHE_DIR}"

python "${PROJECT_ROOT}/train_llm_e2e.py" \
  --design_base_dir /home/user/zedongpeng/workspace/Huggingface/forgehls_lite_10designs \
  --ood_design_base_dir /home/user/zedongpeng/workspace/Huggingface/forgehls_benchmark \
  --output_dir "${OUTPUT_DIR}" \
  --cache_dir "${CACHE_DIR}" \
  --model_name /home/user/zedongpeng/workspace/GiT/zedong/Code-Verification/Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --max_length 1024 \
  --embed_batch_size 12 \
  --batch_size 1024 \
  --epochs 30 \
  --patience 8 \
  --learning_rate 1e-3 \
  --weight_decay 1e-2 \
  --warmup_ratio 0.05 \
  --layer_probe_fractions 0.6,0.7,0.8,0.9 \
  --seed 42 \
  "$@"
