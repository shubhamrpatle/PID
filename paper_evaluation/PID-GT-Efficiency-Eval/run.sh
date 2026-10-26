#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-all}"
if (( $# )); then shift; fi
if [[ "$MODE" == --help || "$MODE" == -h ]]; then
  printf '%s\n' 'Usage: bash run.sh [all|smoke|full] [--max-gpus 4] [--gpus 0 1 2 3 4 5 6 7] [--output DIR]'
  exit 0
fi
PYTHON_BIN="${PYTHON_BIN:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}}"
if [[ -z "$PYTHON_BIN" && -x "$ROOT/../.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT/../.venv/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || true)}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  printf '%s\n' 'Activate the existing evaluation environment or set PYTHON_BIN=/absolute/path/to/python.' >&2
  exit 2
fi
export PYTHON_BIN
case "$MODE" in
  all)
    "$PYTHON_BIN" -u "$ROOT/remote.py" --smoke "$@"
    exec "$PYTHON_BIN" -u "$ROOT/remote.py" "$@"
    ;;
  smoke) exec "$PYTHON_BIN" -u "$ROOT/remote.py" --smoke "$@" ;;
  full) exec "$PYTHON_BIN" -u "$ROOT/remote.py" "$@" ;;
  *) printf '%s\n' 'First argument must be all, smoke, or full.' >&2; exit 2 ;;
esac
