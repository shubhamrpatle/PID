#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == --help ]]; then
  printf '%s\n' 'Usage: MAX_WORKERS=4 bash run.sh [candidate GPU indices, default 0..7]' \
    'Reuses the existing environment, PID checkpoint and frozen paired-evaluation plan.' \
    'Runs a cache-stability gate, then both address modes on DIOR and DOTAv2.'
  exit 0
fi
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
  elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
  elif [[ -x "$ROOT/../.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT/../.venv/bin/python"
  else
    PYTHON_BIN="$(command -v python3 || command -v python || true)"
  fi
fi
PYTHON_BIN="$(command -v "$PYTHON_BIN" || true)"
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" || -d "$PYTHON_BIN" ]]; then
  printf '%s\n' 'Activate the existing evaluation environment or set PYTHON_BIN to its Python executable.' >&2
  exit 2
fi
export PYTHON_BIN
export RS_PROTOCOL_HF_HOME="${RS_PROTOCOL_HF_HOME:-$ROOT/.hf-cache}"
if (( $# == 0 )); then set -- 0 1 2 3 4 5 6 7; fi
exec "$PYTHON_BIN" -u "$ROOT/launch.py" --gpus "$@" --max-workers "${MAX_WORKERS:-4}"
