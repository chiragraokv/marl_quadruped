#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 PATH_TO_CHECKPOINT [additional play.py options]" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

CHECKPOINT="$1"
shift
if [[ ! -x "$REPO_ROOT/.venv/bin/python" ]]; then
  echo "Missing $REPO_ROOT/.venv; create it before playback." >&2
  exit 1
fi

DEVICE_ARGS=()
case "${MASQ_DEVICE:-auto}" in
  auto) ;;
  cpu)
    export CUDA_VISIBLE_DEVICES=""
    DEVICE_ARGS=(--device cpu)
    ;;
  cuda|gpu)
    export CUDA_VISIBLE_DEVICES="${MASQ_GPU_IDS:-0}"
    DEVICE_ARGS=(--device cuda:0)
    ;;
  cuda:*)
    DEVICE_ARGS=(--device "$MASQ_DEVICE")
    ;;
  *)
    echo "MASQ_DEVICE must be auto, cpu, gpu, cuda, or cuda:N" >&2
    exit 2
    ;;
esac

XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_ROOT/.cache}" \
MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/mpl}" \
"$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/scripts/play.py" Unitree-Go2-MASQ-Flat \
  --checkpoint-file "$CHECKPOINT" \
  "${DEVICE_ARGS[@]}" \
  "$@"
