#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 PATH_TO_CHECKPOINT [additional play.py options]" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

CHECKPOINT="$1"
shift
source .venv/bin/activate

XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/laserhammer-cache}" \
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}" \
python scripts/play.py Unitree-Go2-MASQ-Flat \
  --checkpoint-file "$CHECKPOINT" \
  "$@"
