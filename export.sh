#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/common.sh"
if (( $# != 2 )); then
  printf '%s\n' 'Usage: bash export.sh /path/to/best.pt /path/to/new-model-folder' >&2
  exit 2
fi
exec "$PYTHON_BIN" -m refwave.export --model "${MODEL_DIR:-$WORK_ROOT/LocateAnything-3B}" \
  --checkpoint "$1" --output "$2"
