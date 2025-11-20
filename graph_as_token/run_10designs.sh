#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/env/bin/python}"
SCRIPT_PATH="${SCRIPT_DIR}/train_e2e.py"

KERNEL_BASE_DIR="${KERNEL_BASE_DIR:-/home/user/zedongpeng/workspace/HLS-Perf-Prediction-with-GNNs/Graphs/forgehls_kernels/kernels}"
DESIGN_BASE_DIR="${DESIGN_BASE_DIR:-/home/user/zedongpeng/workspace/Huggingface/forgehls_lite_10designs}"
OOD_DESIGN_BASE_DIR="${OOD_DESIGN_BASE_DIR:-/home/user/zedongpeng/workspace/Huggingface/forgehls_benchmark}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/delta_e2e/graph_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/runs/forgehls_10designs}"

BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-80}"
LR="${LR:-1e-3}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
NUM_LAYERS="${NUM_LAYERS:-3}"
DROPOUT="${DROPOUT:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
VALID_RATIO="${VALID_RATIO:-0.1}"
SEED="${SEED:-42}"
MAX_PAIRS="${MAX_PAIRS:-0}"
DEVICE="${DEVICE:-cuda}"
GRAPH_AS_TOKEN="${GRAPH_AS_TOKEN:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"

mkdir -p "${OUTPUT_DIR}"

CMD=(
  "${PYTHON_BIN}" "${SCRIPT_PATH}"
  --kernel_base_dir "${KERNEL_BASE_DIR}"
  --design_base_dir "${DESIGN_BASE_DIR}"
  --ood_design_base_dir "${OOD_DESIGN_BASE_DIR}"
  --cache_root "${CACHE_ROOT}"
  --output_dir "${OUTPUT_DIR}"
  --batch_size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --hidden_dim "${HIDDEN_DIM}"
  --num_layers "${NUM_LAYERS}"
  --dropout "${DROPOUT}"
  --weight_decay "${WEIGHT_DECAY}"
  --train_ratio "${TRAIN_RATIO}"
  --valid_ratio "${VALID_RATIO}"
  --seed "${SEED}"
  --max_pairs "${MAX_PAIRS}"
  --device "${DEVICE}"
  --num_workers "${NUM_WORKERS}"
)

if [[ "${GRAPH_AS_TOKEN}" == "1" ]]; then
  CMD+=(--graph_as_token)
fi

CMD+=("$@")

set -x
"${CMD[@]}"
set +x
