#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -x "$REPO_ROOT/.venv/bin/python" ]]; then
  echo "Missing $REPO_ROOT/.venv; create it before training." >&2
  exit 1
fi

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_ROOT/.cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/mpl}"
GPU_IDS="${MASQ_GPU_IDS:-0}"
case "${GPU_IDS,,}" in
  cpu|none)
    GPU_ARG="None"
    ;;
  all)
    GPU_ARG="all"
    ;;
  *)
    # Tyro requires brackets for the list branch of list[int] | Literal | None.
    GPU_ARG="[$GPU_IDS]"
    ;;
esac

exec "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/scripts/train.py" \
  Unitree-Go2-MASQ-Flat \
  --env.scene.num-envs="${MASQ_NUM_ENVS:-4096}" \
  --gpu-ids "$GPU_ARG" "$@"
