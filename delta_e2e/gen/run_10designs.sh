#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Helper launcher for the diffusion-based graph generator on the ForgeHLS
# 10-design subset. This simplified script uses fixed defaults (edit below)
# and forwards any extra CLI arguments to the Python entry point.
#
# Example:
#   ./run_10designs.sh --eval_batches 4
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODULE_PATH="delta_e2e.gen.train_graph_diffusion"

# Fixed configuration (edit these as needed)
PYTHON_BIN="python"
KERNEL_BASE_DIR="/home/user/zedongpeng/workspace/HLS-Perf-Prediction-with-GNNs/Graphs/forgehls_kernels/kernels"
DESIGN_BASE_DIR="/home/user/zedongpeng/workspace/Huggingface/forgehls_lite_10designs"
OOD_DESIGN_BASE_DIR="/home/user/zedongpeng/workspace/Huggingface/forgehls_benchmark"
CACHE_ROOT="${REPO_ROOT}/delta_e2e/graph_cache"

DEVICE="0"
BATCH_SIZE="1"
EPOCHS="60"
LR="1e-4"
HIDDEN_DIM="32"
TIME_EMBED_DIM="96"
TIMESTEPS="750"
MAX_PAIRS="128"  # set to 0 to use every available pair
LOADER_WORKERS="0"
NODE_LOSS_WEIGHT="1.0"
EDGE_LOSS_WEIGHT="1.0"
SEED="42"
EVAL_BATCHES="0"
EDGE_THRESHOLD="0.55"
MODE="joint"
LATENT_DIM="64"

mkdir -p "${SCRIPT_DIR}/runs"
mkdir -p "${CACHE_ROOT}"

# Run from repo root so the module path resolves without PYTHONPATH hacks
cd "${REPO_ROOT}"

set -x
"${PYTHON_BIN}" -m "${MODULE_PATH}" \
  --kernel_base_dir "${KERNEL_BASE_DIR}" \
  --design_base_dir "${DESIGN_BASE_DIR}" \
  --cache_root "${CACHE_ROOT}" \
  --ood_design_base_dir "${OOD_DESIGN_BASE_DIR}" \
  --device "${DEVICE}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --hidden_dim "${HIDDEN_DIM}" \
  --time_embed_dim "${TIME_EMBED_DIM}" \
  --timesteps "${TIMESTEPS}" \
  --loader_workers "${LOADER_WORKERS}" \
  --node_loss_weight "${NODE_LOSS_WEIGHT}" \
  --edge_loss_weight "${EDGE_LOSS_WEIGHT}" \
  --seed "${SEED}" \
  --eval_batches "${EVAL_BATCHES}" \
  --edge_threshold "${EDGE_THRESHOLD}" \
  --max_pairs "${MAX_PAIRS}" \
  --mode "${MODE}" \
  --latent_dim "${LATENT_DIM}" \
  --amp \
  --nodes_only \
  "$@"
set +x
