#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Quick launcher for ProgSG multimodal training on the ForgeHLS 100-design set.
#
# Usage:
#   ./run_100designs.sh [extra-python-args]
# ---------------------------------------------------------------------------

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$(command -v python)"
SCRIPT_PATH="${PROJECT_ROOT}/train_progsg_e2e.py"

DESIGN_BASE_DIR="/home/user/zedongpeng/workspace/Huggingface/forgehls_lite_100designs"
OOD_DESIGN_BASE_DIR="/home/user/zedongpeng/workspace/Huggingface/forgehls_benchmark"
OUTPUT_DIR="${PROJECT_ROOT}/progsg_runs/forgehls_100designs"
CACHE_ROOT="${PROJECT_ROOT}/progsg_cache"
MAX_DESIGNS=0  # 0 = use every available design
DEVICE="cuda"
BATCH_SIZE=32 # no effect on OOM
EPOCHS=200
LR="4e-4"
LR_SCHEDULER="plateau"
LR_MIN="1e-6"
LR_WARMUP_EPOCHS=20
LR_WARMUP_START_FACTOR="0.1"
LR_PLATEAU_PATIENCE=15
LR_PLATEAU_FACTOR="0.5"
HIDDEN_DIM=128 # OOM
NUM_LAYERS=2 # OOM
HEADS=8
NUM_WORKERS=4
SEED=42
TRAIN_RATIO=0.7
VALID_RATIO=0.2
CODE_ENCODER_NAME="Salesforce/codet5-small"
CODE_MAX_LENGTH=64
CODE_CACHE_DEVICE="cuda"
CODE_TRANSFORMER_LAYERS=4
CODE_TRANSFORMER_HEADS=8
CODE_FUSION_MODE="concat"

mkdir -p "${OUTPUT_DIR}"

CMD=(
  "${PYTHON_BIN}" "${SCRIPT_PATH}"
  --design_base_dir "${DESIGN_BASE_DIR}"
  --ood_design_base_dir "${OOD_DESIGN_BASE_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --cache_root "${CACHE_ROOT}"
  --device "${DEVICE}"
  --batch_size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --lr_scheduler "${LR_SCHEDULER}"
  --lr_min "${LR_MIN}"
  --lr_warmup_epochs "${LR_WARMUP_EPOCHS}"
  --lr_warmup_start_factor "${LR_WARMUP_START_FACTOR}"
  --lr_plateau_patience "${LR_PLATEAU_PATIENCE}"
  --lr_plateau_factor "${LR_PLATEAU_FACTOR}"
  --hidden_dim "${HIDDEN_DIM}"
  --num_layers "${NUM_LAYERS}"
  --heads "${HEADS}"
  --num_workers "${NUM_WORKERS}"
  --seed "${SEED}"
  --train_ratio "${TRAIN_RATIO}"
  --valid_ratio "${VALID_RATIO}"
  --code_encoder_name "${CODE_ENCODER_NAME}"
  --code_max_length "${CODE_MAX_LENGTH}"
  --code_cache_device "${CODE_CACHE_DEVICE}"
  --code_transformer_layers "${CODE_TRANSFORMER_LAYERS}"
  --code_transformer_heads "${CODE_TRANSFORMER_HEADS}"
  --code_fusion_mode "${CODE_FUSION_MODE}"
  --code_local_files_only
  --use_swanlab
)

if [[ "${MAX_DESIGNS}" != "0" ]]; then
  CMD+=(--max_designs "${MAX_DESIGNS}")
fi

CMD+=("$@")

set -x
"${CMD[@]}"
set +x
