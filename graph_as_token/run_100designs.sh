#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# -----------------------------------------------------------------------------
# Configuration (edit as needed)
# -----------------------------------------------------------------------------
KERNEL_BASE_DIR="${KERNEL_BASE_DIR:-/home/user/zedongpeng/workspace/HLS-Perf-Prediction-with-GNNs/Graphs/forgehls_kernels/kernels}"
DESIGN_BASE_DIR="${DESIGN_BASE_DIR:-/home/user/zedongpeng/workspace/Huggingface/forgehls_lite_100designs}"
OOD_DESIGN_BASE_DIR="${OOD_DESIGN_BASE_DIR:-/home/user/zedongpeng/workspace/Huggingface/forgehls_benchmark}"
CACHE_ROOT="${CACHE_ROOT:-/home/user/zedongpeng/workspace/HLS-Perf-Prediction-with-GNNs/delta_e2e/graph_cache}"
OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-output/qor_e2e_run}"

EPOCHS=${EPOCHS:-120}
BATCH_SIZE=${BATCH_SIZE:-16}
LEARNING_RATE=${LEARNING_RATE:-1e-3}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
HIDDEN_DIM=${HIDDEN_DIM:-256}
NUM_LAYERS=${NUM_LAYERS:-3}
DROPOUT=${DROPOUT:-0.1}
GNN_TYPE=${GNN_TYPE:-gcn}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
TRAIN_RATIO=${TRAIN_RATIO:-0.8}
VALID_RATIO=${VALID_RATIO:-0.1}
SEED=${SEED:-42}
MAX_PAIRS=${MAX_PAIRS:-0}
DEVICE=${DEVICE:-cuda}
NUM_WORKERS=${NUM_WORKERS:-0}
USE_SWANLAB=${USE_SWANLAB:-1}
REBUILD_CACHE=${REBUILD_CACHE:-0}

DATE_SUFFIX=$(date +"%y-%m-%d-%H-%M")
OUTPUT_DIR="${OUTPUT_DIR_BASE}_${DATE_SUFFIX}"

GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

if [[ ! -d "${DESIGN_BASE_DIR}" ]]; then
  echo -e "${YELLOW}${BOLD}Warning:${NC} design base directory not found: ${DESIGN_BASE_DIR}"
fi

mkdir -p "${OUTPUT_DIR}"

CMD=(
  "${PYTHON_BIN:-$(which python)}" "${PWD}/train_e2e.py"
  --kernel_base_dir "${KERNEL_BASE_DIR}"
  --design_base_dir "${DESIGN_BASE_DIR}"
  --ood_design_base_dir "${OOD_DESIGN_BASE_DIR}"
  --cache_root "${CACHE_ROOT}"
  --output_dir "${OUTPUT_DIR}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --lr "${LEARNING_RATE}"
  --weight_decay "${WEIGHT_DECAY}"
  --hidden_dim "${HIDDEN_DIM}"
  --num_layers "${NUM_LAYERS}"
  --dropout "${DROPOUT}"
  --gnn_type "${GNN_TYPE}"
  --grad_accum_steps "${GRAD_ACCUM_STEPS}"
  --max_grad_norm "${MAX_GRAD_NORM}"
  --train_ratio "${TRAIN_RATIO}"
  --valid_ratio "${VALID_RATIO}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --num_workers "${NUM_WORKERS}"
)

if [[ "${MAX_PAIRS}" != "0" ]]; then
  CMD+=(--max_pairs "${MAX_PAIRS}")
fi
if [[ "${USE_SWANLAB}" == "1" ]]; then
  CMD+=(--use_swanlab)
fi
if [[ "${REBUILD_CACHE}" == "1" ]]; then
  CMD+=(--rebuild_cache)
fi

color_msg() {
  local color="$1"; shift
  echo -e "${color}${BOLD}$*${NC}"
}

color_msg "${BLUE}" "🚀 Launching QoR training"
color_msg "${GREEN}" "Output directory: ${OUTPUT_DIR}"

set -x
"${CMD[@]}"
set +x
