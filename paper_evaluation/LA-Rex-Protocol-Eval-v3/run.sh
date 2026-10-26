#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || true)}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  printf '%s\n' 'Activate the existing evaluation environment or set PYTHON_BIN=/absolute/path/to/python.' >&2
  exit 2
fi
export PYTHON_BIN
MODE="${1:-all}"
shift "$(( $# > 0 ? 1 : 0 ))"
case "$MODE" in
  -h|--help) printf '%s\n' 'Usage: bash run.sh [all|smoke|full] [--output DIR] [--max-gpus 1-4]'; exit 0 ;;
  smoke) exec "$PYTHON_BIN" -u "$ROOT/remote.py" --smoke "$@" ;;
  full) exec "$PYTHON_BIN" -u "$ROOT/remote.py" "$@" ;;
  all)
    "$PYTHON_BIN" -u "$ROOT/remote.py" --smoke "$@"
    exec "$PYTHON_BIN" -u "$ROOT/remote.py" "$@"
    ;;
  *) printf '%s\n' 'Usage: bash run.sh [all|smoke|full] [--output DIR] [--max-gpus 1-4]' >&2; exit 2 ;;
esac
