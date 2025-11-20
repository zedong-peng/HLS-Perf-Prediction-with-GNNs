#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

python -m baseline.codeT5.run \
  --metric lut \
  --epochs 1 \
  --batch-size 2 \
  --learning-rate 1e-4 \
  --train-ratio 0.6 \
  --val-ratio 0.2 \
  --max-designs 5 \
  --max-ood-designs 5 \
  --device cpu \
  --no-swanlab \
  "$@"
