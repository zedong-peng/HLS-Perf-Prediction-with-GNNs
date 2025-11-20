#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Launch inference for a trained ProgSG model on ForgeHLS 100-design + OOD set.
#
# Usage:
#   ./inference_100designs.sh [overrides...]
# Any extra arguments are forwarded to inference_progsg_e2e.py.
# ---------------------------------------------------------------------------

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$(command -v python)"
SCRIPT_PATH="${PROJECT_ROOT}/inference_progsg_e2e.py"

DESIGN_BASE_DIR="/home/user/zedongpeng/workspace/Huggingface/forgehls_lite_100designs"
OOD_DESIGN_BASE_DIR="/home/user/zedongpeng/workspace/Huggingface/forgehls_benchmark"
TRAIN_RUN_DIR="${PROJECT_ROOT}/progsg_runs/forgehls_100designs"
MODEL_PATH="${TRAIN_RUN_DIR}/checkpoints/model_last.pt"
OUTPUT_DIR="${TRAIN_RUN_DIR}/inference_last"
CACHE_ROOT="${PROJECT_ROOT}/progsg_cache"
DEVICE="cuda"
BATCH_SIZE=32
NUM_WORKERS=4

mkdir -p "${OUTPUT_DIR}"

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "ERROR: Expected checkpoint at ${MODEL_PATH} but it does not exist." >&2
  echo "Run ./run_100designs.sh first or adjust MODEL_PATH." >&2
  exit 1
fi

# Reuse caches from training to avoid reparsing during inference.
if compgen -G "${CACHE_ROOT}/ood/design_cache_*" > /dev/null; then
  for cache_dir in "${CACHE_ROOT}"/ood/design_cache_*; do
    cache_name="$(basename "${cache_dir}")"
    target="${CACHE_ROOT}/${cache_name}"
    [[ -e "${target}" ]] || ln -s "${cache_dir}" "${target}"
  done
fi

if compgen -G "${CACHE_ROOT}/design_cache_*" > /dev/null; then
  mkdir -p "${CACHE_ROOT}/id"
  for cache_dir in "${CACHE_ROOT}"/design_cache_*; do
    cache_name="$(basename "${cache_dir}")"
    target="${CACHE_ROOT}/id/${cache_name}"
    [[ -e "${target}" ]] || ln -s "${cache_dir}" "${target}"
  done
fi

CMD=(
  "${PYTHON_BIN}" "${SCRIPT_PATH}"
  --model_path "${MODEL_PATH}"
  --ood_design_base_dir "${OOD_DESIGN_BASE_DIR}"
  --id_design_base_dir "${DESIGN_BASE_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --cache_root "${CACHE_ROOT}"
  --device "${DEVICE}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
)

CMD+=("$@")

set -x
"${CMD[@]}"
set +x
