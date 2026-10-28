#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

source .venv/bin/activate

XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/laserhammer-cache}" \
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}" \
python scripts/train.py Unitree-Go2-MASQ-Flat \
  --env.scene.num-envs="${MASQ_NUM_ENVS:-4096}" \
  --gpu-ids "${MASQ_GPU_IDS:-all}" \
  "$@"
