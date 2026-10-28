#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

source "$REPO_ROOT/.venv/bin/activate"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_ROOT/.cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/mpl}"
if [[ "${MASQ_GPU_IDS:-0}" == "all" ]]; then
  python scripts/train.py Unitree-Go2-MASQ-Flat \
    --env.scene.num-envs="${MASQ_NUM_ENVS:-4096}" \
    --gpu-ids all "$@"
else
  # Keep tyro on its default single-GPU selection; visibility controls which
  # physical devices are available and also supports comma-separated IDs.
  export CUDA_VISIBLE_DEVICES="${MASQ_GPU_IDS:-0}"
  echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  if [[ "${MASQ_GPU_IDS:-0}" == *,* ]]; then
    python scripts/train.py Unitree-Go2-MASQ-Flat \
      --env.scene.num-envs="${MASQ_NUM_ENVS:-4096}" \
      --gpu-ids all "$@"
  else
    python scripts/train.py Unitree-Go2-MASQ-Flat \
      --env.scene.num-envs="${MASQ_NUM_ENVS:-4096}" \
      "$@"
  fi
fi
